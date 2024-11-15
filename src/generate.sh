#!/bin/bash

# datasets=( "reddit" "uci" "wikipedia" )
datasets=("lastfm" "mooc")
# "enron" "mooc" "Contacts" "CanParl" "Flights"  "SocialEvo" "UNtrade" "lastfm" "UNvote" )
S=10  # number of samples

for dataset in "${datasets[@]}" 
do
    echo "Processing dataset: $dataset"

    for (( j=1; j<=S; j++ )) 
    do
        # python intensify.py -d "$dataset" -m 5 -s $j
        python shuffle.py -d "$dataset" -s $j
    done
done
