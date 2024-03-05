#!/bin/bash
total_missed_values=$((0))
for ls_filename in $(ls test-images/*-*.jpg);
do
    given_filename=$(echo $ls_filename | cut -d'/' -f2);
    given_filename=$(echo $given_filename | cut -d'.' -f1);
    output_groundtruth_name=$(echo "test-images/"$given_filename"_groundtruth.txt");
    echo Running $ls_filename on grade.py;
    python3 grade.py $ls_filename output.txt;
    missed_values=$(diff -y --suppress-common-lines output.txt $output_groundtruth_name | wc -l)
    total_missed_values=$(($total_missed_values + $missed_values));
done

echo Total number of values missed was: $total_missed_values;
