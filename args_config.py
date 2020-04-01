#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-12 19:52:41
@LastEditTime: 2019-08-23 14:16:48
'''

from utils import ArgParser
argparser = ArgParser()
argparser.add_arg("--use_cuda", type=int, default=1, help="Use CUDA")
argparser.add_arg("--gpu_id", type=int, default=1, help="GPU ID")
argparser.add_arg("--deterministic", type=int, default=1,
                  help="Whether fix random seed")
argparser.add_arg("--n_thread", type=int, default=1,
                  help="Number of threads used in multithreading functions")
argparser.add_arg("--model", type=str,
                  default="model_mask_usv", help="Model name")
argparser.add_arg("--dataset", type=str,
                  default="mnist", help="Dataset name")
argparser.add_arg("--save_best_model_k", type=int,
                  default="1", help="Save best k models")
argparser.add_arg("--checkpoint_dir", type=str,
                  default="model", help="Checkpoint sub-directory under ./checkpoint")
argparser.add_arg("--model_comment", type=str, default="",
                  help="Model comment as suffix")
argparser.add_arg("--full_model_comment", type=str, default="",
                  help="Full model comment as suffix")
argparser.add_arg("--batch_size", type=int, default=32, help="Mini-batch size")
argparser.add_arg("--img_height", type=int, default=7, help="Image Height")
argparser.add_arg("--img_width", type=int, default=7, help="Image Width")
argparser.add_arg("--n_class", type=int, default=10,
                  help="Number of categories")
argparser.add_arg("--in_channels", type=int, default=3,
                  help="Number of input channels")
argparser.add_arg('--hidden_list', nargs='+', type=int, default=[40], help="Network hidden layer configuration")
argparser.add_arg('--block_list', nargs='+', type=int, default=[8, 10], help="Network hidden layer block size configuration")
argparser.add_arg('--kernel_list', nargs='+', type=int, default=[32, 32], help="Network conv layer channel configuration")
argparser.add_arg('--pool_out_size', type=int, default=5, help="AdaptivePooling Output Size")
argparser.add_arg("--act", type=str, default="relu",
                  help="Activation function")
argparser.add_arg("--act_thres", type=float, default=4,
                  help="Activation clipping threshold")
argparser.add_arg("--lr", type=float, default=0.001,
                  help="Initial learning rate")
argparser.add_arg("--lr_gamma", type=float, default=0.92,
                  help="Learning rate decay rate")
argparser.add_arg("--lambda_1", type=float, default=10,
                  help="Lambda coefficient for quantization loss")
argparser.add_arg("--lambda_2", type=float, default=0.1,
                  help="Lambda coefficient for lipschitz loss")
argparser.add_arg("--lambda_4", type=float, default=0.1,
                  help="Lambda coefficient for weight l2-norm regularization loss")
argparser.add_arg("--lambda_5", type=float, default=0.0001,
                  help="Lambda coefficient for sigma l1-norm regularization loss")
argparser.add_arg("--lambda_6", type=float, default=0.02,
                  help="Lambda coefficient for weight group lasso loss")
argparser.add_arg("--lambda_7", type=float, default=0.02,
                  help="Lambda coefficient for protective group lasso loss")
argparser.add_arg('--lipschitz_exclude_layers', nargs='*', type=str, default=[], help="Network hidden layers that are excluded from lipschitz constraint")
argparser.add_arg("--cross_layer_lipschitz", type=int, default=0,
                  help="Whether add cross-layer lipschitz loss")
argparser.add_arg("--conditional_update", type=int, default=0,
                  help="Whether perform conditional update on voltages")
argparser.add_arg("--voltage_decay_rate", type=float, default=0.00005,
                  help="Voltage decay rate")
argparser.add_arg("--weight_decay_rate", type=float, default=0.0005,
                  help="Weight decay rate")
argparser.add_arg("--epoch", type=int, default=61,
                  help="Number of epoch for training")
argparser.add_arg("--phase_1", type=int, default=40,
                  help="End epoch of phase 1")
argparser.add_arg("--phase_2", type=int, default=43,
                  help="End epoch of phase 2")
argparser.add_arg("--max_retrain_epoch", type=int, default=3,
                  help="Maximum retraining epoch per quantization")
argparser.add_arg("--log_interval", type=int,
                  default=100, help="Log print interval")
argparser.add_arg("--train_quantize", type=int, default=1,
                  help="Whether apply quantization to training")
argparser.add_arg("--val_quantize", type=int, default=1,
                  help="Whether apply quantization to testing")
argparser.add_arg("--strict_quantize_mask", type=int, default=1,
                  help="Whether aplly strict quantization mask")
argparser.add_arg("--quantize_interval", type=int,
                  default=1, help="Quantization interval in unit of epoch")
argparser.add_arg("--quantize_delay", type=int,
                  default=50000, help="Quantization delay step")
argparser.add_arg("--quantize_bit", type=int,
                  default=8, help="Quantization bits")
argparser.add_arg("--voltage_clamp_per", type=float,
                  default=1, help="Voltage clamp percentile")
argparser.add_arg("--v_pi", type=float, default=4.36,
                  help="Voltage to cause pi phase lag")
argparser.add_arg("--v_max", type=float, default=10.8,
                  help="Maximum voltage that the voltage supply can offer")
argparser.add_arg("--gamma_noise_std", type=float, default=0.002,
                  help="Standard deviation of gamma noise")
argparser.add_arg("--phase_noise_std", type=float, default=0.02,
                  help="Standard deviation of phase noise")
argparser.add_arg("--disk_noise_std", type=float, default=0.02,
                  help="Standard deviation of disk transmitivity noise")
argparser.add_arg("--phase_noise_protect_rate", type=float, default=0.8,
                  help="Protection rate of phase noise")
argparser.add_arg("--train_noise", type=int, default=0,
                  help="Whether add gamma noise to training")
argparser.add_arg("--train_clip_voltage", type=int, default=0, help="Whether clip voltage in training")
argparser.add_arg("--train_clip_upper_perc", type=float, default=1, help="clip voltage upper percentage")
argparser.add_arg("--weight_bit", type=int,
                  default=32, help="Quantization bits")
argparser.add_arg("--input_bit", type=int,
                  default=32, help="Quantization bits")
argparser.add_arg("--mode", type=str,
                  default="oconv", help="Engine Mode")
argparser.add_arg("--train_mode", type=str,
                  default="noise", help="Training Mode, unaware, noise, calibrate")
argparser.add_arg("--out_par", type=int,
                  default=1, help="output channel parallelism")
argparser.add_arg("--batch_par", type=int,
                  default=1, help="batch-wise parallelism")


args = argparser.parse_args()
print(args)
argparser.print_args()
