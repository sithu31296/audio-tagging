from torch.optim.lr_scheduler import StepLR, MultiStepLR


__all__ = {
    "steplr": StepLR
}

def get_scheduler(cfg, optimizer):
    scheduler_name = cfg['SCHEDULER']['NAME']
    assert scheduler_name in __all__.keys(), f"Unavailable scheduler name >> {scheduler_name}.\nList of available schedulers: {list(__all__.keys())}"
    return __all__[scheduler_name](optimizer, *cfg['SCHEDULER']['PARAMS'])


if __name__ == '__main__':
    import torch
    model = torch.nn.Linear(1024, 50)
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    scheduler = MultiStepLR(optimizer, [10, 20, 30], gamma=0.5)