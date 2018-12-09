:: convert all notebooks to markdown
for %%i in (..\..\notebooks\*.ipynb) do jupyter nbconvert --output-dir ..\markdown_notebooks\ --to markdown --template markdown.tpl %%i

:: add YAML preambles to all md converted notebooks
for %%i in (..\markdown_notebooks\*.md) do python nbmd.py %%i