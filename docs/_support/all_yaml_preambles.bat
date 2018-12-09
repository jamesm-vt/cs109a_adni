:: add YAML preambles to all md converted notebooks
for %%i in (markdown_notebooks\*.md) do python _support/nbmd.py %%i