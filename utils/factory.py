# Baseline

def get_model(method, args):
    method = method.lower()

    if method == "onlyshared":
        from methods.rsk import SplitLoRAV6
        return SplitLoRAV6(args)   
    else:
        raise ValueError(f"Unknown method: {method}")




