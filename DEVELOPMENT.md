## Pypi build

First create a pypi account. Instructions here.

Install build
```
python3 -m pip install --upgrade build
```

Specify version if it needs to be incremented in Setup.py

Run the following command from the directory where setup.py is located (this should be scSHARP_tool per git hub repo name)

```
python3 -m build
```

Install/ upgrade twine:
```
python3 -m pip install --upgrade twine
```

```
twine upload <path to tarball>
```

OR if updating existing pypi package

```
twine upload --skip-existing <path to tarball>
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

