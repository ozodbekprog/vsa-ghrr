"""gpuvsa — GHRR / HeRR / FHRR kernels for the DGX Spark retrofit study."""
try:  # the compiled C++ core is optional for the pure-PyTorch parts
    from .vsacore_py import FHRR, GHRR, HeRR, capacity_fhrr, capacity_ghrr
    __all__ = ["FHRR", "GHRR", "HeRR", "capacity_fhrr", "capacity_ghrr"]
except ImportError:  # pragma: no cover
    FHRR = GHRR = HeRR = None
    capacity_fhrr = capacity_ghrr = None
    __all__ = []

__version__ = "0.1.0"
