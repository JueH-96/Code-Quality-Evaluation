import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
indegree = [0] * (N + 1)
for _ in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    index += 2
    indegree[B] += 1
candidates = []
for i in range(1, N + 1):
    if indegree[i] == 0:
        candidates.append(i)
if len(candidates) == 1:
    print(candidates[0])
else:
    print(-1)