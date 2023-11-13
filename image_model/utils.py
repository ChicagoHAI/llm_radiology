from dataclasses import dataclass, field
import argparse
import configparser
import datetime
import os
import torch
import h5py
import numpy as np



def save_preprocessed_data(images, txt_labels, image_labels, preprocessed_data_path):
    os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)

    with h5py.File(preprocessed_data_path, 'w') as f:
        images_group = f.create_group('images')
        txt_labels_group = f.create_group('text_labels')
        img_labels_group = f.create_group('image_labels')
        for idx, image in enumerate(images):
            images_group.create_dataset(str(idx), data=image.numpy())
            txt = txt_labels[idx]
            label = image_labels[idx]
            txt_labels_group.create_dataset(str(idx), data=txt.numpy())
            img_labels_group.create_dataset(str(idx), data=label.numpy())
            # labels_group.create_dataset(str(idx), data=image_labels[idx].numpy())
        # for idx, (image, (txt, label)) in enumerate(zip(images)):
        #     images_group.create_dataset(str(idx), data=image.numpy())


def load_preprocessed_data(preprocessed_data_path):
    images = []
    image_labels = []
    txt_labels = []

    with h5py.File(preprocessed_data_path, 'r') as f:
        images_group = f['images']
        txt_labels_group = f['text_labels']
        img_labels_group = f['image_labels']
        # labels_group = f['labels']

        for idx in range(len(images_group.keys())):
            image = torch.tensor(images_group[str(idx)])
            # txt, label = [torch.tensor(x) for x in labels_group[str(idx)]]
            images.append(image)

            txt = torch.tensor(txt_labels_group[str(idx)])
            label = torch.tensor(img_labels_group[str(idx)])
            txt_labels.append(txt)
            image_labels.append(label)

            # image_labels.append((txt, label))

    return images, image_labels, txt_labels


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""
    
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
    
    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :] 
    
    return img


class Args(object):
    def __init__(self, contain=None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name[-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if not (value is None) or (name[:2] == "__" and name[-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        # give args elements dictionary order to ensure its replicate-ability
        return sorted(list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__)).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)

class String(object):
    @staticmethod
    def to_basic(string):
        """
        Convert the String to what it really means.
        For example, "true" --> True as a bool value
        :param string:
        :return:
        """
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except ValueError:
                pass
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False
        else:
            return string.strip("\"'")  # for those we want to add space before and after the string

@dataclass
class WrappedSeq2SeqTrainingArguments():
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.
        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    cfg: str = field()
    # sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    # predict_with_generate: bool = field(
    #     default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    # )
    # input_max_length: int = field(
    #     default=1536, metadata={
    #         "help": "The sequence max_length we feed into the model, the rest part will be truncated and dropped."}
    # )
    # # 1536 is an initial number, which is 512 for description and 1024 for the table(and other kb representation
    # # ) + question.
    # generation_max_length: int = field(
    #     default=512, metadata={
    #         "help": "The max_length to use on each evaluation loop when predict_with_generate=True."
    #                 " Will default to the max_length value of the model configuration."}
    # )
    # generation_num_beams: int = field(
    #     default=4, metadata={
    #         "help": "The num_beams to use on each evaluation loop when predict_with_generate=True."
    #                 " Will default to the num_beams value of the model configuration."}
    # )
    # load_weights_from: Optional[str] = field(
    #     default=None, metadata={
    #         "help": "The directory to load the model weights from."}
    # )


class Configure(object):
    @staticmethod
    def get_file_cfg(file):
        """
        get configurations in file.
        :param file:
        :return: configure args
        """
        cfgargs = Args()
        parser = configparser.ConfigParser()
        parser.read(file)
        for section in parser.sections():
            setattr(cfgargs, section, Args())
            for item in parser.items(section):
                setattr(getattr(cfgargs, section), item[0], String.to_basic(item[1]))
        return cfgargs
    

    @staticmethod
    def refresh_args_by_file_cfg(file, prev_args):
        args = Configure.get_file_cfg(file)
        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        for arg_name, arg in prev_args:
            if arg is None:
                continue
            if arg_name != "cfg":
                names = arg_name.split(".")
                cur = args
                for name in names[: -1]:
                    if getattr(cur, name) is None:
                        setattr(cur, name, Args())
                    cur = getattr(cur, name)
                if getattr(cur, names[-1]) is None:
                    setattr(cur, names[-1], arg)
        return args


    @staticmethod
    def Get(cfg):
        args = Configure.get_file_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, cfg))

        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        return args