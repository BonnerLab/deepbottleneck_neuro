import sys
import os
base_path = os.path.dirname(__file__)
sys.path.append(base_path)
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import ResNetConfig, ResNetModel, ResNetForImageClassification
from typing import List
# from helpers.initialise_models import initialise_pca_model, initialise_random_model
from helpers.initialise_models import initialise_random_model
from main_new_OG import make_pca, make_random
from classifier import Classifier
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers.trainer_utils import get_last_checkpoint


import evaluate
import numpy as np
import torch
import logging
from datasets import load_dataset
# from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    # MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    # AutoConfig,
    # AutoImageProcessor,
    # AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

from transformers import TrainerCallback
import os

class CustomSaveCallback(TrainerCallback):
    def __init__(self, output_dir, save_epochs=[  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  14,  16,
        18,  20,  25,  30,  35,  40,  45,  50,  60,  70,  80,  90, 100,
       110, 120, 130, 140, 150] ):
        super().__init__()
        self.save_epochs = set(save_epochs)  # Convert to a set for quick lookup
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # Check if the current epoch is in the save list
        print(state)
        if state.epoch in self.save_epochs:
            # Save the model checkpoint
            # output_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{int(state.epoch)}")
            # state.save_model(output_dir)
            # print(f"Model checkpoint saved at {output_dir}")
            # TrainerControl
            control.should_save = True
        else:
            control.should_save = False

        




class PcaModelConfig(PretrainedConfig):
    model_type='pca_model'
    def __init__(self,
                 std_path = '/home/robinbs1/cortical_transformers/deepbottleneck/std_coefs',
                 var_path = '/home/robinbs1/cortical_transformers/deepbottleneck/variance-coefs',
                 channel_sizes=[27, 64, 64, 64, 64, 64, 64, 64, 64],
                 random_init = False,
                 static_norm = True,
                 nb_classes = 1000,
                 avg_ker_size=1,
                 avgpool=False,
                 resnet_baseline=False,
                 **kwargs):
        self.std_path=std_path
        self.var_path=var_path
        self.channel_sizes=channel_sizes
        self.random_init=random_init
        self.static_norm=static_norm
        self.nb_classes=nb_classes
        self.avg_ker_size=avg_ker_size
        self.avgpool=avgpool
        self.resnet_baseline=resnet_baseline
        super().__init__(**kwargs)


class PcaModel(PreTrainedModel):
    config_class=PcaModelConfig
    def __init__(self,config):
        super().__init__(config)
        if config.random_init:

            # self.model = initialise_random_model(target_dir=config.std_path)
            self.model = make_random(channel_sizes=config.channel_sizes, 
                                     std_path = config.std_path,
                                     static_norm = config.static_norm)
            print(f"Initialized Random Model, static_norm={config.static_norm}")
        else:

            # self.model=initialise_pca_model(std_target_dir=config.std_target_dir,
                                        # var_target_dir=config.var_target_dir)
            self.model = make_pca(channel_sizes = config.channel_sizes, 
                                  var_path = config.var_path,
                                  std_path = config.std_path,
                                  static_norm = config.static_norm)
            print(f"Initialized PCA Model, static_norm={config.static_norm}")
        self.classifier = Classifier(input_type=self.model[-1].output_type, nb_classes=config.nb_classes,avg_ker_size=config.avg_ker_size,avgpool=config.avgpool)
    # def forward(self, tensor, labels=None):
    #     logits=self.classifier(self.model(tensor))
    #     if labels is not None:
    #         loss = torch.nn.functional.cross_entropy(logits, labels)
    #         return {"loss": loss, "logits": logits}
    #     return {"logits": logits}
    def forward(self, pixel_values, labels=None):
        logits=self.classifier(self.model(pixel_values))
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

AutoConfig.register("pca_model", PcaModelConfig)
AutoModel.register(PcaModelConfig, PcaModel)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    img_size: int = field(
        default=224,
        metadata={"help": "The nunmber of pixels that the input should be"},

    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )
        
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # model_name_or_path: str = field(
    #     default="google/vit-base-patch16-224-in21k",
    #     metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    # )
    # model_type: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    # )
    # config_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # )
    std_path: str = field(
        default='//home/robinbs1/cortical_transformers/deepbottleneck/std_coefs/64-pca',
        metadata={"help": "path to load std coefficients from pre-training, also used in random init."},
    )
    var_path: str = field(
        default='/home/robinbs1/cortical_transformers/deepbottleneck/variance-coefs/64-pca',
        metadata={"help": "path to load var coefficients from pre-trainin, NOT used in random init."},
    )
    # https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    # https://dev.to/devasservice/python-trick-using-dataclasses-with-fielddefaultfactory-4159
    channel_sizes: List[int] = field(
        # default_factory=lambda: [27, 64, 64, 64, 64, 64, 64, 64, 64],
        default_factory=list,
        metadata={"help": "Model Channel Sizes, 64 channel is [27, 64, 64, 64, 64, 64, 64, 64, 64]"},
    )


    random_init: bool = field(
        default=False,
        metadata={"help": "if true, will use a randomly intitialized model."},
    )

    static_norm: bool = field(
        default=True,
        metadata={"help": "if true, uses loaded mean and std for normalization, otherwise uses batchnorm2d."},
    )
    resnet_baseline: bool = field(
        default=False,
        metadata={"help": "if true, uses resnet baseline."},
    )
    freeze_half: bool = field(
        default=False,
        metadata={"help": "if True will freeze <=half of layers."},
    )

    

    # image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

class SimpleImageProcessor():
    def __init__(self,size):
        self.size=dict(height=size,width=size)
        # image_processor.size["height"], image_processor.size["width"]
        

def main(args_json_file=None):




    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if args_json_file is not None:
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_json_file)

    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: this should be extended in the future!
    print(f"model_args={model_args}")
    pca_config = PcaModelConfig(std_path=model_args.std_path,
                                var_path=model_args.var_path,
                                channel_sizes=model_args.channel_sizes,
                                random_init=model_args.random_init,
                                resnet_baseline=model_args.resnet_baseline,
                                static_norm=model_args.static_norm
                                )
    if model_args.resnet_baseline:
        resnet_configuration = ResNetConfig(num_labels=pca_config.nb_classes)
        pca_model = ResNetForImageClassification(resnet_configuration)
        print("Initialized ResNet Model")
    else:
        pca_model = PcaModel(pca_config)
        if model_args.freeze_half:
            n_layers = len(pca_model.model)
            for layer_index, layer in enumerate(pca_model.model):
                if layer_index <= n_layers/2:
                    for param in layer.parameters():
                        param.requires_grad = False  # Freeze the parameters
                    print(f"freezing layer {layer_index}")


    image_processor=SimpleImageProcessor(size=data_args.img_size)

    # parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #         # If we pass only one argument to the script and it's the path to a json file,
    #         # let's parse it to get our arguments.
    #     data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     data_args, training_args = parser.parse_args_into_dataclasses()


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )

    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if data_args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {data_args.image_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--image_column_name` to the correct audio column - one of "
            f"{', '.join(dataset_column_names)}."
        )
    if data_args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(dataset_column_names)}."
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    

    # image_processor = AutoImageProcessor.from_pretrained(
    #         model_args.image_processor_name or model_args.model_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         token=model_args.token,
    #         trust_remote_code=model_args.trust_remote_code,
    #     )

    # # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        dataset["validation"].set_transform(val_transforms)

    # # Initialize our trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"] if training_args.do_train else None,
    #     eval_dataset=dataset["validation"] if training_args.do_eval else None,
    #     compute_metrics=compute_metrics,
    #     processing_class=image_processor,
    #     data_collator=collate_fn,
    # )

    # Initialize our trainer
    trainer = Trainer(
        model=pca_model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        # processing_class=None,
        data_collator=collate_fn,
        # args=TrainingArguments(save_strategy='no'),
        callbacks=[CustomSaveCallback(output_dir=training_args.output_dir)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    
if __name__ == "__main__":
    main()