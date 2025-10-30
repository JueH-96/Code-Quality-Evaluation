n, *rest = map(int, open(0).read().split())
X = rest[:n]

for i in range(n-4, -1, -1):
    if X[i] + X[i+3] < X[i+1] + X[i+2]:
        X[i+1] = X[i] + X[i+3] - X[i+2]
        X[i+2] = X[i] + X[i+3] - X[i+1]

print(sum(X))