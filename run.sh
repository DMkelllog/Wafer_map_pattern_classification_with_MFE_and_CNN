for i in {0..1}
do
  for j in {0..1}
  do
    for k in {0..1}
    do
      python run_model.py $i $j $k
    done
  done
done