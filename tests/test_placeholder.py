def test_imports():
    import pandas as pd
    import numpy as np
    assert pd.__version__ is not None
    assert np.__version__ is not None

def test_structure():
    import os
    assert os.path.exists('data')
    assert os.path.exists('notebooks')
    assert os.path.exists('tests')