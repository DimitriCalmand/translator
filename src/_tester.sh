nb_test=0
function tester
{
    nb_test=$(($nb_test+1))
    nb_encoder=$1
    nb_decoder=$2
    nb_heads=$3
    embed_dim=$4
    feed_forward_dim=$5
    max_lenght=$6
    num_words=$7
    echo -e "nb_encoder=$nb_encoder\nnb_decoder=$nb_decoder\nnb_heads=$nb_heads\nembed_dim=$embed_dim\nfeed_forward_dim=$feed_forward_dim\nmax_lenght=$max_lenght\nnum_words=$num_words" > _config_test.py
    return_value=$(./test.py test)
    config_test=$(cat _config_test.py)
    echo -ne "$config_test\n$return_value" > tests/test"$nb_test"
    echo "finish test$nb_test"
}
tester 1 1 1 128 100 30 3000
tester 1 1 1 128 200 30 3000
tester 1 1 1 32 100 30 3000
tester 4 1 1 128 100 30 3000
tester 1 4 1 128 100 30 3000
tester 1 1 8 128 100 30 3000
tester 1 1 1 256 100 30 3000
tester 2 2 1 128 100 30 3000
tester 4 1 4 128 100 30 3000
tester 1 4 8 128 100 30 3000

