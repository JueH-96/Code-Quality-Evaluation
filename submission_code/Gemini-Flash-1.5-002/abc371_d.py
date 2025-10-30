# YOUR CODE HERE
N = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    L, R = map(int, input().split())
    total_villagers = 0
    for i in range(N):
        if L <= X[i] <= R:
            total_villagers += P[i]
    print(total_villagers)