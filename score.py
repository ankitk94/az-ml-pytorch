import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
from azureml.core.model import Model

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def build_model():
    path = Model.get_model_path('model.pth')
    model = torch.load(path)
    model.eval()
    return model

def init():
    global model
    model = build_model()

def run(raw_data):
    import json
    data = json.loads(raw_data)['data']
    data = np.array(data)
    var = Variable(torch.from_numpy(data).float())
    result = model(var).data
    resultList = np.array(result.numpy()).tolist()
    return json.dumps({"result": resultList})
