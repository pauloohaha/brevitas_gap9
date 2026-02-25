import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import shape_inference, helper, numpy_helper
import copy

from devo_onnx.extractor import BasicEncoder4Evs_noskip

if __name__ == "__main__":
    
    evs_bins = 5
    match_feat_dim = 64
    ctx_feat_dim = 96
    dim = 32

    onnx_fnet = BasicEncoder4Evs_noskip(
                bins=evs_bins,
                output_dim=match_feat_dim,       #MATCHING FEATURE DIMENSION
                dim=dim,
                norm_fn="batch",
    )
    onnx_fnet.eval()

    onnx_inet = BasicEncoder4Evs_noskip(
                bins=evs_bins,
                dim=dim,
                output_dim=ctx_feat_dim,       #CONTEXT FEATURE DIMENSION
                norm_fn="none",
    )
    onnx_inet.eval()
    trained_weight = torch.load('../TinyDEVO_batchnorm.pth', weights_only=False, map_location=torch.device('cpu'))

    for name, param in onnx_inet.named_parameters():
        param.data = trained_weight['model_state_dict']['patchify.ctx_feat_encoder.' + name]

    for name, param in onnx_fnet.named_parameters():
        param.data = trained_weight['model_state_dict']['patchify.matching_feat_encoder.' + name]

    for name, buf in onnx_fnet.named_buffers():
        buf.data = trained_weight['model_state_dict']['patchify.matching_feat_encoder.' + name]
    
    input_output = np.load('deeploy/deeploy_ifmap.npz')

    test_image    = torch.from_numpy(input_output['events']).float()
    ref_test_fmap = torch.from_numpy(input_output['fmap']).float()
    ref_test_imap = torch.from_numpy(input_output['imap']).float()

    np.savez('deeploy/inet/inputs.npz',
             images=test_image[0].cpu().numpy()
             )
    print("Reshaped input saved to deeploy/inet/inputs.npz")
    
    np.savez('deeploy/fnet/inputs.npz',
             images=test_image[0].cpu().numpy()
             )
    print("Reshaped input saved to deeploy/fnet/inputs.npz")

    test_imap = onnx_inet(test_image) / 4.0
    test_fmap = onnx_fnet(test_image) / 4.0
    print("imap max diff: " + str(torch.max(test_imap - ref_test_imap[0])))
    print("fmap max diff: " + str(torch.max(test_fmap - ref_test_fmap[0])))
    np.savez('deeploy/inet/outputs.npz',
             imap=test_imap.detach().numpy())
    np.savez('deeploy/fnet/outputs.npz',
             fmap=test_fmap.detach().numpy())
    