import numpy as np
import random


def params_factory_00():
    params = dict(
        alpha=0.3,
        s_max=0.98 + 1,
        height=300000,
        features=[
            dict(
                lift=3,
                linlift=dict(k=1, b=0.5),
                beta=0.2,
            ),
            dict(
                lift=2,
                linlift=dict(k=-1, b=1.5),
                beta=0.1,
            ),
        ]
    )
    for i in range(13):
        params['features'].append(params['features'][1])
    print(len(params['features']))
    return params


def params_factory_01():
    params = dict(
        alpha=0.3,
        s_max=0.98 + 1,
        height=300000,
        features=[
            dict(
                lift=3,
                linlift=dict(k=0.75, b=0.5),
                beta=0.2,
            ),
            dict(
                lift=2,
                linlift=dict(k=-0.75, b=1.25),
                beta=0.1,
            ),
        ]
    )
    for i in range(15):
        params['features'].append(params['features'][(i < 7) * 1])

    return params


def generate_dataset(params):
    random.seed(42)
    np.random.seed(42)
    alpha = params["alpha"]
    h, w = params['height'], len(params['features'])
    label = (np.random.rand(h, 1) < params['alpha']) * 1
    f_time = np.random.rand(h, 1)
    betas = np.array([f["beta"] for f in params['features']])
    # lifts = np.array([f["lift"] for f in params['features']])
    liftk = np.array([f["linlift"]['k'] for f in params['features']])
    liftb = np.array([f["linlift"]['b'] for f in params['features']])
    lifts = liftk * f_time + liftb
    gammas = (1 / (alpha * lifts) - 1) / (1 / alpha - 1) * betas

    s = (np.random.rand(h, 1) < params['s_max']) * 1
    probs = s * (label * betas + (1 - label) * gammas) + (1 - s) * 0.5
    features = np.random.rand(h, w) < probs
    return features, f_time, label


def main():
    params = params_factory_01()
    features, f_time, label = generate_dataset(params)
    np.savez_compressed('pool_cross_00', features=features, f_time=f_time, label=label)


if __name__ == '__main__':
    main()
