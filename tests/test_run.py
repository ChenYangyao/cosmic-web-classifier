import numpy as np
from cosmic_web_classifier import classifiler_from_particles, TidalClassifier
from pyhipp.stats import Rng
import pytest

@pytest.fixture
def rng():
    return Rng(10086)

@pytest.fixture
def ptcls(rng: Rng):
    return rng.uniform(0.0, 500.0, (128**3, 3))

@pytest.fixture
def classifier(ptcls: np.ndarray):
    return classifiler_from_particles(l_box=500.0, n_grids=128, positions=ptcls,
                                      r_sm=10.0, lam_th=0.2)
    
def test_web_types(classifier: TidalClassifier):
    
    assert np.all(classifier.is_filament 
                + classifier.is_sheet
                + classifier.is_void
                + classifier.is_knot 
                == np.ones_like(classifier.is_filament))

def test_get_web_points(classifier: TidalClassifier):
    
    for t in classifier.web_types.keys():
        x = classifier.grid_points_of_web_type(t)
        assert x.shape[1] == 3