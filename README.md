# Graph and distributed extensions of the Douglas–Rachford method: Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* K. Bredies, E. Chenchene, E. Naldi. Graph and distributed extensions of the Douglas–Rachford method. 2022. [ArXiv preprint](https://arxiv.org/abs/2211.04782)

To reproduce the results of the numerical experiments in Section 5, run:
```bash
python3 main.py
```

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{bcn2022,
  author = {Bredies, Kristian and Chenchene, Enis and Naldi, Emanuele},
  title = {Graph and Distributed Extensions of the {D}ouglas-–{R}achford Method},
  journal = {SIAM Journal on Optimization},
  volume = {34},
  number = {2},
  pages = {1569--1594},
  year = {2024}
}
```

## Requirements

Please make sure to have the following Python modules installed, most of which should be standard.

* [numpy>=1.20.1](https://pypi.org/project/numpy/)
* [scipy>=1.6.2](https://pypi.org/project/scipy/)
* [networkx>=1.6.2](https://pypi.org/project/networkx/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [Pillow>=8.2.0](https://pypi.org/project/Pillow/)

## Acknowledgments  

* | ![](<euflag.png>) | This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 861137. |
  |-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

* All data used for numerical experiments in this project have been created artificially by the authors.

## License  
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
