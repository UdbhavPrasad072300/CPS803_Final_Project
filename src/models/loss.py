import torch.nn as nn


class Soft_Distillation_Loss(nn.Module):
    def __init__(self, lambda_balancing: float, temperature):
        super(Soft_Distillation_Loss, self).__init__()

        self.lambda_balancing = lambda_balancing
        self.temperature = temperature

        self.CE_student = nn.CrossEntropyLoss()
        self.KLD_teacher = nn.KLDivLoss()

    def forward(self, teacher_y, student_y, y):

        loss = ((1-self.lambda_balancing) * self.CE_student(student_y, y)) + \
               (self.lambda_balancing * (self.temperature**2) *
                self.KLD_teacher(student_y / self.temperature, teacher_y / self.temperature))

        return loss


class Hard_Distillation_Loss(nn.Module):
    def __init__(self):
        super(Hard_Distillation_Loss, self).__init__()

        self.CE_teacher = nn.CrossEntropyLoss()
        self.CE_student = nn.CrossEntropyLoss()

    def forward(self, teacher_y, student_y, y):
        
        loss = (1/2) * (self.CE_student(student_y, y)) + (1/2) * (self.CE_teacher(student_y, teacher_y.argmax(dim=-1)))

        return loss
