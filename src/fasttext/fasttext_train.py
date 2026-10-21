import fasttext
from src.fasttext.fasttext_data_load import data_load_path,get_save_model_path


def train(autotune, split_mode):
    train_dir = data_load_path(is_train=True, split_mode=split_mode)
    dev_dir = data_load_path(is_dev=True, split_mode=split_mode)
    test_dir = data_load_path(is_test=True, split_mode=split_mode)
    if autotune:
        fasttext_model = fasttext.train_supervised(train_dir,
                                                   autotuneValidationFile=dev_dir,
                                                   autotuneDuration=60,
                                                   )
    else:
        fasttext_model = fasttext.train_supervised(train_dir, )

    fasttext_model.save_model(get_save_model_path(autotune=autotune, split_mode=split_mode))
    print(fasttext_model.test(test_dir))
    print(f'split_mode={split_mode}, autotune={autotune}')


def main():
    autotune_list = [False, True]
    split_modes = ['char','jieba']
    for autotune in autotune_list:
        for split_mode in split_modes:
            train(autotune, split_mode)


if __name__ == '__main__':
    main()
