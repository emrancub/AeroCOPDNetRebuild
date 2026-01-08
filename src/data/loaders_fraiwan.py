# Fraiwan uses the same LungSoundDataset class (full recording rows).
# Windowing is handled in manifest or via preprocessing choices; for simplicity, treat full recordings as one segment.
from .loaders_icbhi import LungSoundDataset
