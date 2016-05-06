#!/bin/sh

sphinx-apidoc -o api --separate -H BNAS -A "Robert Östling" ../bnas
sphinx-build -b html . html
