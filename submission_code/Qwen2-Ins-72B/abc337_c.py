from collections import deque
N = int(input())
A = list(map(int, input().split()))
ans = deque()
for i in range(N):
    if A[i] == -1:
        ans.appendleft(i+1)
    else:
        A[A[i]-1] = -1
print(*ans)