import torch
import torch.nn as nn


class MOEScore(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,noise_gating=True,num_experts=8,k=4,aux_loss_weight=0.01):
        super(MOEScore,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.noise_gating=noise_gating
        self.num_experts=num_experts
        self.k=k
        self.aux_loss_weight=aux_loss_weight

        self.router=nn.Linear(input_size,num_experts)

        self.experts=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size,hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size,hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size,output_size),
                nn.Sigmoid()
            ) for _ in range(num_experts)
        ])

        with torch.no_grad():
            self.router.weight.data.normal_(0.0,0.1)
        
        if self.noise_gating:
            self.noise_eps=1e-10
            self.noise_std=1.0

        self.register_buffer('experts_counts',torch.zeros(num_experts))
        self.total_samples=0


    def cv_squared(self,x):
        eps=1e-2
        mean=torch.mean(x)
        variance=torch.mean(torch.square(x-mean))
        return variance/(mean**2+eps)
    
    def compute_prob(self,x):
        """ 
        x:[batch_size,input_size]
        """
        router_logits=self.router(x)#[batch_size,num_experts]
        if self.noise_gating and self.training:
            noise=torch.randn_like(router_logits)*self.noise_std
            router_logits+=noise
        
        router_probs=torch.softmax(router_logits,dim=-1)
        return router_logits,router_probs
    
    def forward(self,x):
        batch_size=x.size(0)

        router_logits,router_probs=self.compute_prob(x)
        top_k_probs,top_k_idx=torch.topk(router_probs,self.k,dim=-1)#[batch_size,self.k]

        top_k_normal=top_k_probs/top_k_probs.sum(dim=-1,keepdim=True)

        importance=router_probs.sum(dim=0)#[num_experts]
        self.experts_counts+=importance
        self.total_samples+=batch_size

        aux_loss=self.cv_squared(importance)+self.cv_squared(router_probs.mean(dim=0))

        final_output=torch.zeros(batch_size,self.output_size,device=x.device)

        for expert_idx in range(self.num_experts):
            indicates=(expert_idx==top_k_idx).nonzero(as_tuple=True)
            if len(indicates[0])>0:
                batch_row,batch_col=indicates

                expert_input=x[batch_row]
                expert_output=self.experts[expert_idx](expert_input)

                for i,(row,col) in enumerate(zip(batch_row,batch_col)):
                    weight=top_k_normal[row,col]
                    final_output[row]+=weight*expert_output[i]
        
        return final_output,aux_loss
    

class MOETrain:
    def __init__(self,model,learning_rate=1e-3):
        self.model=model
        self.task_loss=nn.MSELoss()
        self.optim=torch.optim.AdamW(model.parameters(),lr=learning_rate)
        self.history={'total_loss':[],'task_loss':[],'aux_loss':[]}

    def train_step(self,x,y,aux_loss_weight=None):
        self.model.train()
        self.optim.zero_grad()
        y_pred,aux_loss=self.model(x)
        task_loss=self.task_loss(y_pred,y)

        weight=aux_loss_weight if aux_loss_weight is not None else self.model.aux_loss_weight
        total_loss=task_loss+aux_loss*weight

        total_loss.backward()
        self.optim.step()

        self.history['total_loss'].append(total_loss.item())
        self.history['task_loss'].append(task_loss.item())
        self.history['aux_loss'].append(aux_loss.item())

        return task_loss.item(),total_loss.item(),aux_loss.item()
    
    def train(self,train_loader,epochs=10,aux_loss_weight=None):
        #从模型中自动获取device参数
        device=next(self.model.parameters()).device
        for epoch in range(epochs):
            epoch_total=0
            epoch_task=0
            epoch_aux=0

            for x,y in train_loader:
                x,y=x.to(device),y.to(device)
                task_loss,total_loss,aux_loss=self.train_step(x,y)

                epoch_total+=total_loss
                epoch_aux+=aux_loss
                epoch_task+=task_loss

            n=len(train_loader)
            mean_total=epoch_total/n
            mean_task=epoch_task/n
            mean_aux=epoch_aux/n

            print(f'total_loss averge:{mean_total},task_loss averge:{mean_task},aux_loss averge:{mean_aux}')
            
    def evaluate(self,test_loader):
        self.model.eval()

        total_loss=0

        with torch.no_grad():
            for batch_x,batch_y in test_loader:
                y_pred,_=self.model(batch_x)
                
                task_loss=self.task_loss(y_pred,batch_y)
                total_loss+=task_loss.item()

        n=len(test_loader)
        return total_loss/n
    

if __name__=='__main__':
    pass




                

                

                


        


        
