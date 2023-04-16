import os, sys
sys.path.append(os.path.abspath('/Users/maxperozek/CP499/scSHARP_tool'))
# sys.path.append(os.path.relpath('/Users/maxperozek/CP499/scSHARP_tool'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scSHARP'
copyright = '2023, Daniel Lewinsohn, Max Perozek, William Holtz, Ben Modlin'
author = 'Daniel Lewinsohn, Max Perozek, William Holtz, Ben Modlin'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxdoc'
html_static_path = ['_static']
