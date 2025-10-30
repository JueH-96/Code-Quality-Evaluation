# YOUR CODE HERE
from sys import stdin

N = int(stdin.readline())
X = list(map(int, stdin.readline().split()))
Q = int(stdin.readline())
TG = [list(map(int, stdin.readline().split())) for _ in range(Q)]

X.sort()
TG.sort(key=lambda x: x[1])

total_movements = 0
for i in range(Q):
    T, G = TG[i]
    total_movements += abs(X[T-1] - G)
    X[T-1] = G

print(total_movements)