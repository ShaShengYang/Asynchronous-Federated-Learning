import model

n = model.ResNet18()
total = sum(p.numel() for p in n.parameters())
print(f'Total parameters:{total / 1e6}')
