# python_utils

a few small tools that I greatly rely on in the following area:
- robotics
- computer vision
- machine learning

### file structure

```
├── README.md
├── requirements.txt
└── src
    ├── examples
    │   └── demo_<tools>.ipynb
    ├── unittest
    │   └──  test_<tools>.py
    └── <tools>.py
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
- [ ] PCD: Implement pcd writer
