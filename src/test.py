#!/usr/bin/python3

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def main():
    args = sys.argv[1]
    if args == 'encode':
        import encode
        encode.main()
    elif args == 'main':
        import main
        main.main()
    elif args == 'trash':
        import trash
        trash.main()
    elif args == 'tf_encode':
        import tensorflow_encode as te
        te.main() 
    elif args == 'train':
        import train_tokenizer  
        pass
main()
