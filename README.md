# python_utils

a few small tools that I greatly rely on in the following area:
- robotics
- computer vision
- machine learning

### file structure

```
python_utils
├── examples
│   └── demo_<tool>.ipynb
│
├── src
│   └── py_utils
│       ├── __init__.py
│       └── <tool>>.py
|
├── test
|   ├── __init__.py
│   └── test_<tools>>.py
│
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

### Install

```bash
$ cd <your workspace>
$ git clone https://github.com/ChadLin9596/python_utils
$ cd python_utils
$ pip install -e .
```

### Unittest

```bash
$ cd <directory of this repository>
$ python -m unittest
```

### Maintenance

```bash
$ cd <directory of this repository>
$ python -m black . --line-length=79

# sort all imports in alphabetical order
$ python -m isort .
```

### TODO:
- [ ] add github workflow
- [ ] Data Repo: functionality of attributes serialization
- [ ] Data Repo: implement tools to manage data repo tree
- [x] PCD: Implement pcd writer
