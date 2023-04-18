scSHARP Installation
====================


Github
------

For further installation information visit our Github repositories for the `scSHARP tool <https://github.com/mperozek11/scSHARP_tool>`_ and `R tools <https://github.com/W-Holtz/R4scSHARP>`_

Also see our distribution on `pypi <https://pypi.org/project/scSHARP/>`_

Quickstart
----------

scSHARP and its dependencies can be installed in a few simple steps:

R package installation
**********************

Our package utilizes a suite of cutting edge single cell classification tools which we have compiled as an R package. Please install by following instructions `here <https://github.com/W-Holtz/R4scSHARP>`_.

Note: the R tools are necessary for running scSHARP

Create a conda environment:
***************************

``conda create -n <env name> python=3.9``

``conda activate <env name>``

Torch cluster
*************

Torch cluster needs to be installed separately.

Installing with Pip:
^^^^^^^^^^^^^^^^^^^^
``pip install torch-cluster``

Installing with conda
^^^^^^^^^^^^^^^^^^^^^
``conda install -c ostrokach-forge torch-cluster``

scSHARP package install
***********************

Installing with Pip:
^^^^^^^^^^^^^^^^^^^^
Install with Pip if you plan on running R4scSHARP from Python.

``pip install scSHARP``

``conda install pytorch-cluster -c pyg``


Installing with conda
^^^^^^^^^^^^^^^^^^^^^
Install with Conda if you do not plan on running R4scSHARP from Python.

``conda install -c lewinsohndp scsharp -c pyg -c conda-forge -c r``

