from federatedscope.register import register_trainer
from federatedscope.core.trainers import BaseTrainer
import torch
import torch.optim as optim
from torch.ao.quantization import prepare_qat


class QuantTrainer(BaseTrainer):
    def __init__(self, model, data, device, **kwargs):
        # NN modules
        super().__init__(model, data, device, **kwargs)
        self.model = model
        # FS `ClientData` or your own data
        self.data = data
        # Device name
        self.device = device
        # kwargs
        self.kwargs = kwargs
        # Criterion & Optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # _hook_on_fit_start_init
        self.model.to(self.device)
        # **调用fuse_model进行层融合**
        # if hasattr(self.model, 'fuse_model'):
        #     self.model.fuse_model()
        self.model.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.FakeQuantize.with_args(observer=torch.ao.quantization.MinMaxObserver,
                                                                    qscheme=torch.per_tensor_affine),
            weight=torch.ao.quantization.default_weight_fake_quant
        )
        self.model.train()
        self.model = prepare_qat(self.model, inplace=True)


        total_loss = num_samples = 0
        # _hook_on_batch_start_init
        for inputs, labels in self.data['train']:
            # _hook_on_batch_forward
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # _hook_on_batch_backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # _hook_on_batch_end
            total_loss += loss.item() * labels.shape[0]
            num_samples += labels.shape[0]

        # _hook_on_fit_end
        return num_samples, self.model.cpu().state_dict(), \
            {'loss_total': total_loss, 'avg_loss': total_loss/float(
                num_samples)}

    def evaluate(self, target_data_split_name='test'):
        with torch.no_grad():
            self.model.to(torch.device("cpu:0"))
            self.model.eval()
            total_loss = num_samples = 0
            # _hook_on_batch_start_init
            for inputs, labels in self.data[target_data_split_name]:
                # _hook_on_batch_forward
                inputs, labels = inputs.to(torch.device("cpu:0")), labels.to(torch.device("cpu:0"))
                pred = self.model(inputs)
                loss = self.criterion(pred, labels)

                # _hook_on_batch_end
                total_loss += loss.item() * labels.shape[0]
                num_samples += labels.shape[0]

            # _hook_on_fit_end
            return {
                f'{target_data_split_name}_loss': total_loss,
                f'{target_data_split_name}_total': num_samples,
                f'{target_data_split_name}_avg_loss': total_loss /
                float(num_samples)
            }

    def update(self, model_parameters, strict=False):
        self.model.load_state_dict(model_parameters, strict)

    def get_model_para(self):
        return self.model.cpu().state_dict()


def call_quant_trainer(trainer_type):
    if trainer_type == 'quant_trainer':
        return QuantTrainer


register_trainer('quant_trainer', call_quant_trainer)