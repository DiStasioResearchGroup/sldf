# SLDF - Semi-Local Density Fingerprints

A Python package for generating Semi-Local Density Fingerprints (SLDF).

## Installation

```bash
git clone https://github.com/DiStasioResearchGroup/sldf.git
cd sldf
pip install .
```

## Quick Start

```python
import sldf

rho = electron_density_array      # Total electron density 
s = reduced_gradient_array        # Reduced density gradient for total electron density
weights = integration_weights     # Grid integration weights
nsp = 20                          # Number of B-splines

# Generate SLDF fingerprints
SLDF = sldf.calc_SLDF(rho, s, weights, nsp)
```

## Examples

See the `examples/` directory for integration with:
- PySCF: `examples/pyscf_example/`
- Psi4: `examples/psi4_example/`

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7

## License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

MIT
