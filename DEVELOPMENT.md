## Pypi build

First create a pypi account. Instructions here.

Install/ upgrade twine:
```
python3 -m pip install --upgrade twine
```

```
twine upload <path to tarball>
```

Enter credentials as follows:
```
username: __token__
password: token from pypi api key (this should include the 'pypi-' prefix)
```



## Run sphinx build

From scSHARP_tool directory run:
```
sphinx-build -b html docs/source/ docs/build/html
```

