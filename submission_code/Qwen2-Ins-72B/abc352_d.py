from collections import deque
N, K = map(int, input().split())
P = list(map(int, input().split()))
Q = deque()
ans = float('inf')
for i in range(N):
    while Q and P[Q[-1]] > P[i]:
        Q.pop()
    Q.append(i)
    if i >= K-1:
        if i-K >= Q[0]:
            Q.popleft()
        ans = min(ans, P[Q[0]]+K-1-P[i])
print(ans)