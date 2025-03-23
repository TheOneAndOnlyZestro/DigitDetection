import torch
from simpleModel import Simple

#define hyperparams
learning_rate = 0.01
#define our data
x_train = torch.tensor([[1],[2],[3],[4],[5],[6],[7]], dtype=torch.float32)
y_train = (x_train * 5) + 6

m = Simple()

loss = torch.nn.MSELoss()

optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate)

for epoch in range(300):
    y_hats = m(x_train)

    l = loss(y_hats, y_train)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(f"EPOCH {epoch}: Current loss is: {l}")


print(f"Prediction for 10 is {m(torch.tensor([[10]], dtype=torch.float32 )).item(): .3f}")

