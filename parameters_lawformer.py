import argparse


def parse():
    parser = argparse.ArgumentParser()

    # decimal
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--small_sample_size", type=int, default=10)
    parser.add_argument("--fact_sentence_length", type=int, default=512)
    parser.add_argument("--charge_sentence_length", type=int, default=100)
    parser.add_argument("--article_sentence_length", type=int, default=150)
    parser.add_argument("--min_bound", type=float, default=0.0)
    parser.add_argument("--max_bound", type=float, default=100000000.0)
    # parser.add_argument("--embedding_dims", type=int, default=200)

    # string
    parser.add_argument("--visible_devices", type=str, default="0,1")
    parser.add_argument("--visible_cuda_devices", type=str, default="0,1")
    parser.add_argument("--log_dir", type=str, default="log_lawformer")
    parser.add_argument("--train_datapath", type=str, default="./data/tokenized_data_lawformer/small_train.pkl")
    parser.add_argument("--valid_datapath", type=str, default="./data/tokenized_data_lawformer/small_valid.pkl")
    parser.add_argument("--test_datapath", type=str, default="./data/tokenized_data_lawformer/small_test.pkl")
    parser.add_argument("--model_savepath", type=str, default="./pts/tca_ljp_model_lawformer")
    parser.add_argument("--model_dir", type=str, default="./pts/")
    # parser.add_argument("--embedding_path", type=str, default="./data/additional_data/cail_thulac.npy")

    # boolean
    parser.add_argument("--small_sample", action="store_true")

    args = parser.parse_args()
    return args