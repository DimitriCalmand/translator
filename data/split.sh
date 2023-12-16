#!/bin/sh

if [ $# -eq 4 ]
then
    # Définir le nom du fichier d'entrée
    input_file=$1
    
    # Définir le ratio (par exemple, 0.8 pour 80% et 0.2 pour 20%)
    ratio=$4
    
    # Calculer le nombre de lignes à conserver dans le premier fichier
    total_lines=$(cat $input_file | wc -l)
    split_point=$(echo "$total_lines * $ratio" | bc)
    split_point_rounded=${split_point%.*}
    
    # Créer les deux fichiers
    head -n $split_point_rounded $input_file > $2 
    tail -n +$(($split_point_rounded+1)) $input_file > $3 

    sed -i '1s/^/en,fr\n/' $3
    
    echo "Fichiers divisés en fonction du ratio $ratio."
    echo "Premier fichier contient les premiers $split_point_rounded lignes et est nommé train.csv."
    echo "Deuxième fichier contient le reste des lignes et est nommé test.csv."
fi
