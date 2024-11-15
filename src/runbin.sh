#!/bin/bash

# Run the Python script 50 times
for i in {1..20}
do
    echo "Running iteration $i..."
    python binseqtest.py --m 16
done