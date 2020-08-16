from datetime import datetime
import os
import shutil
import torch


def save_model_checkpoint(state, is_best, output_dir, exp_name, step, model_name):
    file_name = "{}_ckpt_{}.tar".format(model_name, step)
    output_path = os.path.join(output_dir, exp_name, file_name)
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, output_path)
    if is_best:
        best_output_path = os.path.join(output_dir, exp_name, "{}_ckpt_best.tar".format(model_name))
        shutil.copyfile(output_path, best_output_path)
        return best_output_path
    return output_path


def get_exp_name(task_name, model_name):
    return "{}-{}-{}".format(task_name, model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


def eval_and_save(model, exp_name, run_config, epoch, validate_evaluator, best_evaluator, best_output_path_dicts,
                  force_best=None):
    if type(model) != dict:
        best_output_path_dicts = {"model": best_output_path_dicts}
        model_dict = {"model": model}
    else:
        model_dict = model

    if (force_best is not None and force_best) \
            or (validate_evaluator is not None
                and validate_evaluator.is_better_than(best_evaluator, run_config.metrics)):
        print("Best evaluator is updated.")
        best_evaluator = validate_evaluator
        for model_name, model_instance in model_dict.items():
            best_output_path = save_model_checkpoint(state=dict(
                epoch=epoch,
                state_dict=dict([(key, value.to("cpu")) for key, value in model_instance.state_dict().items()]),
            ),
                is_best=True,
                output_dir=run_config.ckpt_dir,
                exp_name=exp_name,
                step=epoch,
                model_name=model_name
            )
            best_output_path_dicts[model_name] = best_output_path
    if epoch % run_config.save_every_epoch_num == 0:
        print("Saving model checkpoint.")
        for model_name, model_instance in model_dict.items():
            save_model_checkpoint(state=dict(
                epoch=epoch,
                state_dict=dict([(key, value.to("cpu")) for key, value in model_instance.state_dict().items()]),
            ),
                is_best=False,
                output_dir=run_config.ckpt_dir,
                exp_name=exp_name,
                step=epoch,
                model_name=model_name
            )
    if type(model) != dict:
        return best_evaluator, best_output_path_dicts["model"]
    else:
        return best_evaluator, best_output_path_dicts
