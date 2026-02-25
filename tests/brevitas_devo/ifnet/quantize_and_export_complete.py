#!/usr/bin/env python3
"""
Comprehensive IFNet Quantization Script

This script performs the complete quantization workflow:
1. Load pretrained FP32 models (fnet and inet)
2. Run FP32 inference and compute SNR vs reference
3. Quantize models to INT8 using Brevitas
4. Calibrate quantized models
5. Run INT8 inference and compute SNR vs reference
6. Export quantized models to ONNX format

Usage:
    python quantize_and_export_complete.py

Output:
    - All outputs saved to quantized_output/ directory
    - FP32 ONNX models
    - INT8 ONNX models with quantized weights
    - SNR comparison report
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
import json

# Add the ifnet directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devo_onnx.extractor import BasicEncoder4Evs_noskip

# Brevitas imports
from brevitas.graph.calibrate import calibration_mode
from brevitas.export.inference import quant_inference_mode
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model


def compute_snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB
    SNR = 10 * log10(P_signal / P_noise)
    """
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean(noise ** 2)
    
    if noise_power == 0 or noise_power.item() == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_metrics(output: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    """
    Compute various metrics between output and reference
    """
    diff = output - reference
    
    metrics = {
        'snr_db': compute_snr(reference, diff),
        'max_diff': torch.max(torch.abs(diff)).item(),
        'mean_diff': torch.mean(torch.abs(diff)).item(),
        'mse': torch.mean(diff ** 2).item(),
        'rmse': torch.sqrt(torch.mean(diff ** 2)).item(),
    }
    
    return metrics


def print_metrics(name: str, metrics: Dict[str, float], indent: int = 0):
    """Print metrics in a formatted way"""
    prefix = "  " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  SNR:        {metrics['snr_db']:.2f} dB")
    print(f"{prefix}  Max Diff:   {metrics['max_diff']:.6f}")
    print(f"{prefix}  Mean Diff:  {metrics['mean_diff']:.6f}")
    print(f"{prefix}  MSE:        {metrics['mse']:.6e}")
    print(f"{prefix}  RMSE:       {metrics['rmse']:.6e}")


def export_onnx_quantized(model: nn.Module, dummy_input: torch.Tensor, 
                          output_path: str, name: str) -> bool:
    """
    Export quantized model to ONNX using quant_inference_mode
    """
    print(f"\n  Exporting {name} to ONNX: {output_path}")
    
    model.eval()
    
    try:
        with torch.no_grad(), quant_inference_mode(model):
            # Run once to cache quantized weights
            _ = model(dummy_input)
            
            # Export to ONNX using legacy exporter
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                opset_version=13,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,  # Use legacy exporter
            )
        
        # Verify the exported model
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"    ✓ Successfully exported ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"    ✗ Export failed: {e}")
        return False


def load_models(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load and initialize fnet and inet models with pretrained weights
    """
    print("="*70)
    print("STEP 1: Loading Models")
    print("="*70)
    
    # Configuration
    evs_bins = 5
    match_feat_dim = 64
    ctx_feat_dim = 96
    dim = 32
    
    # Instantiate networks
    print("\n  Creating model instances...")
    fnet = BasicEncoder4Evs_noskip(
        bins=evs_bins,
        output_dim=match_feat_dim,
        dim=dim,
        norm_fn="batch",
    ).to(device).eval()
    
    inet = BasicEncoder4Evs_noskip(
        bins=evs_bins,
        dim=dim,
        output_dim=ctx_feat_dim,
        norm_fn="none",
    ).to(device).eval()
    
    # Load pretrained weights
    print("  Loading pretrained weights...")
    weight_path = '../TinyDEVO_batchnorm.pth'
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    
    trained_weight = torch.load(weight_path, weights_only=False, map_location=device)
    
    # Load weights for inet
    for name, param in inet.named_parameters():
        full_key = 'patchify.ctx_feat_encoder.' + name
        if full_key in trained_weight['model_state_dict']:
            param.data = trained_weight['model_state_dict'][full_key]
    
    # Load weights for fnet
    for name, param in fnet.named_parameters():
        full_key = 'patchify.matching_feat_encoder.' + name
        if full_key in trained_weight['model_state_dict']:
            param.data = trained_weight['model_state_dict'][full_key]
    
    # Load buffers (BN statistics) for fnet
    for name, buf in fnet.named_buffers():
        full_key = 'patchify.matching_feat_encoder.' + name
        if full_key in trained_weight['model_state_dict']:
            buf.data = trained_weight['model_state_dict'][full_key]
    
    print("  ✓ Models loaded successfully")
    
    return fnet, inet


def run_fp32_evaluation(fnet: nn.Module, inet: nn.Module, 
                        test_input: torch.Tensor, ref_fmap: torch.Tensor, 
                        ref_imap: torch.Tensor) -> Tuple[Dict, Dict]:
    """
    Run FP32 models and compute metrics vs reference
    """
    print("\n" + "="*70)
    print("STEP 2: FP32 Model Evaluation")
    print("="*70)
    
    print(f"\n  Input shape: {test_input.shape}")
    print(f"  Reference fmap shape: {ref_fmap.shape}")
    print(f"  Reference imap shape: {ref_imap.shape}")
    
    # Run inference
    print("\n  Running FP32 inference...")
    with torch.no_grad():
        fp32_fmap = fnet(test_input) / 4.0
        fp32_imap = inet(test_input) / 4.0
    
    # Compute metrics
    fnet_metrics = compute_metrics(fp32_fmap, ref_fmap)
    inet_metrics = compute_metrics(fp32_imap, ref_imap)
    
    # Print results
    print_metrics("fnet (matching feature)", fnet_metrics, indent=1)
    print_metrics("inet (context feature)", inet_metrics, indent=1)
    
    return fnet_metrics, inet_metrics


def quantize_models(fnet: nn.Module, inet: nn.Module, 
                    test_input: torch.Tensor) -> Tuple[nn.Module, nn.Module]:
    """
    Quantize models to INT8 using Brevitas
    """
    print("\n" + "="*70)
    print("STEP 3: Quantizing Models to INT8")
    print("="*70)
    
    print("\n  Quantization configuration:")
    print("    - Weight bit-width: 8")
    print("    - Activation bit-width: 8")
    print("    - Bias bit-width: 32")
    print("    - Weight granularity: per_tensor")
    print("    - Activation type: symmetric")
    print("    - Scale type: float")
    print("    - Backend: layerwise")
    
    # Quantize fnet
    print("\n  Quantizing fnet...")
    quant_fnet = quantize_model(
        model=fnet,
        backend='layerwise',
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=32,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
    )
    
    # Quantize inet
    print("  Quantizing inet...")
    quant_inet = quantize_model(
        model=inet,
        backend='layerwise',
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=32,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
    )
    
    return quant_fnet, quant_inet


def calibrate_models(quant_fnet: nn.Module, quant_inet: nn.Module, 
                     test_input: torch.Tensor):
    """
    Calibrate quantized models
    """
    print("\n" + "="*70)
    print("STEP 4: Calibrating Quantized Models")
    print("="*70)
    
    print("\n  Running calibration...")
    
    with torch.no_grad():
        with calibration_mode(quant_fnet):
            quant_fnet(test_input)
        with calibration_mode(quant_inet):
            quant_inet(test_input)
    
    print("  ✓ Calibration completed")


def run_int8_evaluation(quant_fnet: nn.Module, quant_inet: nn.Module,
                        test_input: torch.Tensor, ref_fmap: torch.Tensor,
                        ref_imap: torch.Tensor) -> Tuple[Dict, Dict]:
    """
    Run INT8 models and compute metrics vs reference
    """
    print("\n" + "="*70)
    print("STEP 5: INT8 Model Evaluation")
    print("="*70)
    
    quant_fnet.eval()
    quant_inet.eval()
    
    # Run inference
    print("\n  Running INT8 inference...")
    with torch.no_grad():
        int8_fmap = quant_fnet(test_input) / 4.0
        int8_imap = quant_inet(test_input) / 4.0
    
    # Compute metrics
    fnet_metrics = compute_metrics(int8_fmap, ref_fmap)
    inet_metrics = compute_metrics(int8_imap, ref_imap)
    
    # Print results
    print_metrics("fnet (matching feature)", fnet_metrics, indent=1)
    print_metrics("inet (context feature)", inet_metrics, indent=1)
    
    return fnet_metrics, int8_fmap, inet_metrics, int8_imap


def export_models(fnet: nn.Module, inet: nn.Module, 
                  quant_fnet: nn.Module, quant_inet: nn.Module,
                  test_input: torch.Tensor, output_dir: str) -> Dict[str, bool]:
    """
    Export all models to ONNX
    """
    print("\n" + "="*70)
    print("STEP 6: Exporting to ONNX")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Export FP32 models
    print("\n  Exporting FP32 models...")
    
    fnet.eval()
    inet.eval()
    
    with torch.no_grad():
        # FP32 fnet
        fp32_path = os.path.join(output_dir, "fnet_fp32.onnx")
        try:
            torch.onnx.export(
                fnet, test_input, fp32_path,
                input_names=['input'], output_names=['output'],
                opset_version=13, do_constant_folding=True,
                export_params=True, dynamo=False,
            )
            size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
            print(f"    ✓ fnet_fp32.onnx ({size_mb:.2f} MB)")
            results['fnet_fp32'] = True
        except Exception as e:
            print(f"    ✗ fnet_fp32.onnx failed: {e}")
            results['fnet_fp32'] = False
        
        # FP32 inet
        inet_path = os.path.join(output_dir, "inet_fp32.onnx")
        try:
            torch.onnx.export(
                inet, test_input, inet_path,
                input_names=['input'], output_names=['output'],
                opset_version=13, do_constant_folding=True,
                export_params=True, dynamo=False,
            )
            size_mb = os.path.getsize(inet_path) / (1024 * 1024)
            print(f"    ✓ inet_fp32.onnx ({size_mb:.2f} MB)")
            results['inet_fp32'] = True
        except Exception as e:
            print(f"    ✗ inet_fp32.onnx failed: {e}")
            results['inet_fp32'] = False
    
    # Export INT8 models
    print("\n  Exporting INT8 models...")
    
    int8_fnet_path = os.path.join(output_dir, "fnet_int8.onnx")
    results['fnet_int8'] = export_onnx_quantized(
        quant_fnet, test_input, int8_fnet_path, "fnet (INT8)"
    )
    
    int8_inet_path = os.path.join(output_dir, "inet_int8.onnx")
    results['inet_int8'] = export_onnx_quantized(
        quant_inet, test_input, int8_inet_path, "inet (INT8)"
    )
    
    return results


def verify_onnx_models(output_dir: str, test_input: np.ndarray):
    """
    Verify exported ONNX models
    """
    print("\n" + "="*70)
    print("STEP 7: Verifying ONNX Models")
    print("="*70)
    
    try:
        import onnx
        import onnxruntime as ort
        
        onnx_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
        
        for model_name in sorted(onnx_files):
            model_path = os.path.join(output_dir, model_name)
            try:
                # Load and check model
                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                
                # Run inference
                sess = ort.InferenceSession(model_path)
                input_name = sess.get_inputs()[0].name
                output = sess.run(None, {input_name: test_input})
                
                print(f"\n  ✓ {model_name}")
                print(f"    Input shape:  {sess.get_inputs()[0].shape}")
                print(f"    Output shape: {output[0].shape}")
                
            except Exception as e:
                print(f"\n  ✗ {model_name} - Verification failed: {e}")
                
    except ImportError:
        print("\n  onnx/onnxruntime not available for verification")


def save_outputs(output_dir: str, int8_fmap: torch.Tensor, int8_imap: torch.Tensor):
    """
    Save quantized outputs for reference
    """
    print("\n  Saving quantized outputs...")
    
    np.savez(
        os.path.join(output_dir, "int8_outputs.npz"),
        fmap=int8_fmap.cpu().numpy(),
        imap=int8_imap.cpu().numpy()
    )
    print("    ✓ int8_outputs.npz saved")


def generate_report(output_dir: str, metrics_fp32: Dict, metrics_int8: Dict,
                    export_results: Dict):
    """
    Generate comprehensive JSON report
    """
    report = {
        'configuration': {
            'weight_bit_width': 8,
            'act_bit_width': 8,
            'bias_bit_width': 32,
            'weight_quant_granularity': 'per_tensor',
            'act_quant_type': 'sym',
            'scale_factor_type': 'float_scale',
            'backend': 'layerwise',
        },
        'fnet': {
            'fp32': metrics_fp32[0],
            'int8': metrics_int8[0],
            'snr_degradation_db': metrics_fp32[0]['snr_db'] - metrics_int8[0]['snr_db'],
        },
        'inet': {
            'fp32': metrics_fp32[1],
            'int8': metrics_int8[1],
            'snr_degradation_db': metrics_fp32[1]['snr_db'] - metrics_int8[1]['snr_db'],
        },
        'export_status': export_results,
    }
    
    report_path = os.path.join(output_dir, 'quantization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also generate text report
    text_report = f"""IFNet Quantization Report
{'='*70}

Configuration:
  Weight bit-width: {report['configuration']['weight_bit_width']}
  Activation bit-width: {report['configuration']['act_bit_width']}
  Bias bit-width: {report['configuration']['bias_bit_width']}
  Granularity: {report['configuration']['weight_quant_granularity']}
  Activation type: {report['configuration']['act_quant_type']}
  Scale type: {report['configuration']['scale_factor_type']}

fnet (Matching Feature Encoder):
  FP32 SNR: {report['fnet']['fp32']['snr_db']:.2f} dB
  INT8 SNR: {report['fnet']['int8']['snr_db']:.2f} dB
  SNR Degradation: {report['fnet']['snr_degradation_db']:.2f} dB
  FP32 Max Diff: {report['fnet']['fp32']['max_diff']:.6f}
  INT8 Max Diff: {report['fnet']['int8']['max_diff']:.6f}

inet (Context Feature Encoder):
  FP32 SNR: {report['inet']['fp32']['snr_db']:.2f} dB
  INT8 SNR: {report['inet']['int8']['snr_db']:.2f} dB
  SNR Degradation: {report['inet']['snr_degradation_db']:.2f} dB
  FP32 Max Diff: {report['inet']['fp32']['max_diff']:.6f}
  INT8 Max Diff: {report['inet']['int8']['max_diff']:.6f}

Export Status:
  fnet_fp32.onnx: {'✓' if export_results['fnet_fp32'] else '✗'}
  inet_fp32.onnx: {'✓' if export_results['inet_fp32'] else '✗'}
  fnet_int8.onnx: {'✓' if export_results['fnet_int8'] else '✗'}
  inet_int8.onnx: {'✓' if export_results['inet_int8'] else '✗'}

Output Directory: {output_dir}
"""
    
    text_path = os.path.join(output_dir, 'quantization_report.txt')
    with open(text_path, 'w') as f:
        f.write(text_report)
    
    print(f"\n  Reports saved:")
    print(f"    - quantization_report.json")
    print(f"    - quantization_report.txt")


def main():
    """
    Main function that orchestrates the complete quantization workflow
    """
    print("\n" + "="*70)
    print("IFNet Complete Quantization Workflow")
    print("="*70)
    
    # Configuration
    device = torch.device('cpu')
    output_dir = "quantized_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load models
        fnet, inet = load_models(device)
        
        # Load test data
        print("\n  Loading test data...")
        input_output_path = 'deeploy/deeploy_ifmap.npz'
        if not os.path.exists(input_output_path):
            raise FileNotFoundError(f"Test data not found: {input_output_path}")
        
        input_output = np.load(input_output_path)
        test_image = torch.from_numpy(input_output['events']).float().to(device)
        ref_fmap = torch.from_numpy(input_output['fmap']).float().to(device)
        ref_imap = torch.from_numpy(input_output['imap']).float().to(device)
        
        # Take first sample if batch dimension differs
        if ref_fmap.shape[0] > 1 and test_image.shape[0] == 1:
            ref_fmap = ref_fmap[0:1]
            ref_imap = ref_imap[0:1]
        
        # Step 2: FP32 evaluation
        fnet_metrics_fp32, inet_metrics_fp32 = run_fp32_evaluation(
            fnet, inet, test_image, ref_fmap, ref_imap
        )
        
        # Step 3: Quantize
        quant_fnet, quant_inet = quantize_models(fnet, inet, test_image)
        
        # Step 4: Calibrate
        calibrate_models(quant_fnet, quant_inet, test_image)
        
        # Step 5: INT8 evaluation
        (fnet_metrics_int8, int8_fmap, 
         inet_metrics_int8, int8_imap) = run_int8_evaluation(
            quant_fnet, quant_inet, test_image, ref_fmap, ref_imap
        )
        
        # Step 6: Export to ONNX
        export_results = export_models(
            fnet, inet, quant_fnet, quant_inet, 
            test_image, output_dir
        )
        
        # Step 7: Verify ONNX
        verify_onnx_models(output_dir, test_image.cpu().numpy())
        
        # Save outputs
        save_outputs(output_dir, int8_fmap, int8_imap)
        
        # Generate report
        generate_report(
            output_dir,
            (fnet_metrics_fp32, inet_metrics_fp32),
            (fnet_metrics_int8, inet_metrics_int8),
            export_results
        )
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"\nOutput directory: {output_dir}/")
        print("\nGenerated files:")
        for f in sorted(os.listdir(output_dir)):
            size = os.path.getsize(os.path.join(output_dir, f))
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.2f} MB"
            print(f"  - {f:<35} {size_str:>10}")
        
        print("\nSNR Comparison:")
        print(f"  fnet: FP32={fnet_metrics_fp32['snr_db']:.2f} dB → INT8={fnet_metrics_int8['snr_db']:.2f} dB "
              f"(Δ={fnet_metrics_fp32['snr_db']-fnet_metrics_int8['snr_db']:.2f} dB)")
        print(f"  inet: FP32={inet_metrics_fp32['snr_db']:.2f} dB → INT8={inet_metrics_int8['snr_db']:.2f} dB "
              f"(Δ={inet_metrics_fp32['snr_db']-inet_metrics_int8['snr_db']:.2f} dB)")
        
        print("\n" + "="*70)
        print("✓ All steps completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print("="*70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
