import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os #saves model

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() #2 linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x): #x is the tensor
        #actuatuon function
        x = F.relu(self.linear1(x))
        x = self.linear2(x) #dont need actuation func
        return x

    def save(self, file_name='model.pth'): #helper func
        model_folder_path = './model'
        if not os.path.exists(model_folder_path): #check if file exisits
            os.makedirs(model_folder_path) #create it 

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma 
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #
        self.criterion = nn.MSELoss() #loss function

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #(n,x)

        if len(state.shape) == 1: #append another dimention for 1 batch
            # (1, x) 1= num of batches
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) #define tuple with 1 val

        # 1: predicted Q values with curr state 
        pred = self.model(state) #3 diff vals

        target = pred.clone() #iterate over tensors and apply formula    pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                 # 2: Q_new = r + y * max(next_predicited Q value) -> only do if not done
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

             # preds[argmax(action)] = Q_new. index of the action is the new q
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad() #empty the gradient
        loss = self.criterion(target, pred)  #(qnew and q)
        loss.backward() #backprop

        self.optimizer.step()


