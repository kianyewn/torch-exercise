import numpy as np
from einops import pack, unpack


def test_einops_unpack():
    """Pack is just concatenate given the specified einops pattern"""
    h, w = 100, 200
    # image_rgb is 3-dimensional (h, w, 3) and depth is 2-dimensional (h, w)
    image_rgb = np.random.random([h, w, 3])
    image_depth = np.random.random([h, w])
    # but we can stack them
    image_rgbd, ps = pack([image_rgb, image_depth], "h w *")

    assert np.allclose(
        np.concatenate([image_rgb, image_depth[:, :, None]], axis=-1), image_rgbd
    )

    assert ps == [(3,), ()] # packed shapes
    

    a, b = unpack(image_rgbd, ps, 'h w *') # torch.Size([16, 2048, 64])
    assert a.shape == (h, w, 3)
    assert b.shape == (h, w)