# YOUR CODE HERE

N, M = map(int, input().split())
A = list(map(int, input().split()))
X = []
for _ in range(N):
    X.append(list(map(int, input().split())))

goal = [min(A) for _ in range(M)]
met = True

for i in range(N):
    for j in range(M):
        if X[i][j] < goal[j]:
            met = False
            break
    if not met:
        break

print('Yes' if met else 'No')