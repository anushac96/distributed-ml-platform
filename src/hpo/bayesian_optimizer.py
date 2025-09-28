import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import scikit-optimize (skopt). If it's not available, fall back to
# a simple random search so the HPO service still runs without the optional
# dependency.
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei
    _SKOPT_AVAILABLE = True
except Exception:
    gp_minimize = None
    Real = Integer = Categorical = None
    gaussian_ei = None
    _SKOPT_AVAILABLE = False
    logger.warning("scikit-optimize (skopt) not available; falling back to random search. Install via 'pip install scikit-optimize' for Bayesian optimization.")

class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(self, search_space: Dict[str, Dict], objective: str, direction: str = "maximize"):
        self.search_space = search_space
        self.objective = objective
        self.direction = direction
        self.dimensions = self._create_dimensions()
        self.parameter_names = list(search_space.keys())
        
        # Storage for observations
        self.X = []  # Parameter combinations
        self.y = []  # Objective values
        
    def _create_dimensions(self) -> List:
        """Convert search space to skopt dimensions"""
        # If skopt isn't installed return an empty dimensions list; the
        # optimizer will fall back to random suggestions.
        if not _SKOPT_AVAILABLE:
            return []

        dimensions = []

        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']

            if param_type == 'float':
                dimensions.append(Real(
                    param_config['min'],
                    param_config['max'],
                    name=param_name
                ))
            elif param_type == 'int':
                dimensions.append(Integer(
                    param_config['min'],
                    param_config['max'],
                    name=param_name
                ))
            elif param_type == 'categorical':
                dimensions.append(Categorical(
                    param_config['choices'],
                    name=param_name
                ))

        return dimensions
    
    async def suggest(self) -> Dict[str, Any]:
        """Suggest next hyperparameter combination"""
        if len(self.X) == 0:
            # First suggestion - random
            return self._random_suggestion()
        # If scikit-optimize isn't available or we don't have valid
        # dimensions, fall back to random suggestions.
        if not _SKOPT_AVAILABLE or not self.dimensions:
            logger.debug("Using random suggestion because skopt is unavailable or no dimensions defined")
            return self._random_suggestion()

        # Use Bayesian optimization via skopt
        try:
            # Convert direction for skopt (it minimizes by default)
            y_values = self.y if self.direction == "minimize" else [-y for y in self.y]

            # Get next suggestion
            result = gp_minimize(
                func=lambda x: 0,  # Dummy function
                dimensions=self.dimensions,
                x0=self.X,
                y0=y_values,
                n_calls=1,
                n_initial_points=0,
                acquisition_func=gaussian_ei
            )

            suggested_point = result.x

        except Exception as e:
            logger.warning(f"Bayesian optimization failed: {e}, using random suggestion")
            return self._random_suggestion()
        
        # Convert to parameter dictionary
        suggestion = {}
        for i, param_name in enumerate(self.parameter_names):
            suggestion[param_name] = suggested_point[i]
        
        logger.info(f"Suggested parameters: {suggestion}")
        return suggestion
    
    def _random_suggestion(self) -> Dict[str, Any]:
        """Generate random suggestion from search space"""
        suggestion = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                value = np.random.uniform(param_config['min'], param_config['max'])
            elif param_type == 'int':
                value = np.random.randint(param_config['min'], param_config['max'] + 1)
            elif param_type == 'categorical':
                value = np.random.choice(param_config['choices'])
            
            suggestion[param_name] = value
        
        return suggestion
    
    async def report(self, parameters: Dict[str, Any], metrics: Dict[str, float]):
        """Report trial results"""
        # Convert parameters to list format
        param_list = [parameters[name] for name in self.parameter_names]
        
        # Get objective value
        if self.objective in metrics:
            objective_value = metrics[self.objective]
            
            self.X.append(param_list)
            self.y.append(objective_value)
            
            logger.info(f"Recorded observation: params={parameters}, objective={objective_value}")
        else:
            logger.warning(f"Objective metric '{self.objective}' not found in metrics: {metrics}")