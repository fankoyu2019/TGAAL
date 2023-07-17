# TGAAL

TGAAL_Recognizer_csORFs.py : input_file="./input/valid_lncRNA.fasta", output_file="./output/TGAAL_RNA_sORFs_result.csv" 
      function Predict_csORFs_by_sORF allows the input of sORF type files to return the encoding potential of each sORF.
      function Predict_csORFs_by_RNA input lncRNA type file will first automatically find all the sORFs fragments and then return the coding potential of each sORF.
GenerateSamples.py : types is the category to be generated. load_path is the path to load the model. file_name is output file. input_dataset_name is input file. generated nums is the number of generated samples.
match_tuples_negative.py: constructing {non-csORF, csORF} tuples.
match_tuples_positive.py: constructing {csORF, non-csORF} tuples.
