class BasicConfig(object):
    def __init__(self, grad_accumulate_steps, max_grad_norm, inform_every_batch_num, save_every_epoch_num,
                 eval_every_epoch_num, use_cuda, learning_rate, adam_epsilon, weight_decay,
                 epoch_num, warmup_steps, metrics, log_dir, ckpt_dir, print_grad_norm, seed, batch_size):
        self.grad_accumulate_steps = grad_accumulate_steps
        self.max_grad_norm = max_grad_norm
        self.inform_every_batch_num = inform_every_batch_num
        self.save_every_epoch_num = save_every_epoch_num
        self.eval_every_epoch_num = eval_every_epoch_num
        self.use_cuda = use_cuda
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay
        self.epoch_num = epoch_num
        self.warmup_steps = warmup_steps
        self.metrics = metrics
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.print_grad_norm = print_grad_norm
        self.seed = seed
        self.batch_size = batch_size

    @staticmethod
    def get_news_graph(epochs=100, use_cuda=True):
        gcn_configs = BasicConfig.get_common(epochs=epochs, use_cuda=use_cuda)
        gcn_configs.eval_every_epoch_num = 50
        gcn_configs.learning_rate = 1e-4
        gcn_configs.weight_decay = 1e-2
        return gcn_configs

    @staticmethod
    def get_common(use_cuda=True, print_grad_norm=False, epochs=100):
        grad_accumulate_steps = 4
        max_grad_norm = None
        # Change this back
        inform_every_batch_num = 40
        # inform_every_batch_num = 500
        save_every_epoch_num = 1
        eval_every_epoch_num = 1
        learning_rate = 1e-3
        adam_epsilon = 1e-8
        weight_decay = 5e-4
        warmup_steps = 0
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        log_dir = "exp_log/"
        ckpt_dir = "exp_ckpt/"
        seed = 42
        batch_size = 32

        return BasicConfig(
            grad_accumulate_steps=grad_accumulate_steps,
            max_grad_norm=max_grad_norm,
            inform_every_batch_num=inform_every_batch_num,
            save_every_epoch_num=save_every_epoch_num,
            eval_every_epoch_num=eval_every_epoch_num,
            use_cuda=use_cuda,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            epoch_num=epochs,
            warmup_steps=warmup_steps,
            metrics=metrics,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            print_grad_norm=print_grad_norm,
            seed=seed,
            batch_size=batch_size
        )
