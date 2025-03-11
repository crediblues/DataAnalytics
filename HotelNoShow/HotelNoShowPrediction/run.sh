#!/bin/bash

# Run the pipeline
python3 src/pipeline.py --db_file data/noshow.db --table_name no_show --model_type logistic_regression
