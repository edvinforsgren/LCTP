#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name=lflc_venv
