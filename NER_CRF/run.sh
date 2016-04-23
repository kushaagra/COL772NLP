python Test_final.py $1 featured_file

java -cp $MALLET_INC cc.mallet.fst.SimpleTagger --model-file crf_trained_original featured_file > my_out

python formatted_output.py $1 $2
