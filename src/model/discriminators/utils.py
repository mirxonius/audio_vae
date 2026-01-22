"""
Utility functions for discriminators.
"""

from typing import Tuple


def get_2d_padding(
    kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)
) -> Tuple[int, int]:
    """Calculate padding for 2D convolution to maintain size"""
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )
