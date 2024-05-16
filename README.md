# Fully Homomorphic Encryption (FHE) and Bootstrapping

This is the code associated to the paper "Fully Homomorphic Encryption and Bootstrapping" by Rémi Leluc, Elie Chedemail, Adéchola Kouande, Quyen Nguyen, Njaka Andriamandratomanana. [PDF](https://hal.science/hal-03676650/document)

## Description

Scripts
- utils.py: contains some tool functions to work on quotient ring polynomial Rq = Zq[X]/(X^N + 1)
- fhe.py: implements functions to perform FHE
- bootstrap.py: implement functions to perform Bootstrap in FHE with KeySwitch and Blind Rotation

Dependencies in Python 3
- requirements.txt : dependencies

## Citation

```bibtex
@phdthesis{leluc2022fully,
  title={Fully homomorphic encryption and bootstrapping},
  author={Leluc, R{\'e}mi and Chedemail, Elie and Kouande, Ad{\'e}chola and Nguyen, Quyen and Andriamandratomanana, Njaka},
  year={2022},
  school={IRMAR-Universit{\'e} Rennes 1}
}
```
