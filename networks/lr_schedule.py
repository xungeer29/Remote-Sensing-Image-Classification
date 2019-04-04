
def half_lr(init_lr, ep):
    lr = init_lr / 2**ep

    return lr

def step_lr(ep):
    if ep < 5:
        lr = 0.005
    elif ep < 10:
        lr = 0.001
    elif ep < 15:
        lr = 0.0005
    elif ep < 20:
        lr = 0.0001
    elif ep < 25:
        lr = 0.00005
    else:
        lr = 0.00001
    return lr
