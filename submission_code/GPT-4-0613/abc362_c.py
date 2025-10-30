import sys
from operator import itemgetter

N = int(sys.stdin.readline())
LR = sorted([list(map(int, sys.stdin.readline().split())) for _ in range(N)], key=itemgetter(1))

left = [0]*(N+1)
right = [0]*(N+1)

for i in range(N):
    left[i+1] = left[i] + LR[i][0]
    right[i+1] = right[i] + LR[i][1]

if max(left) <= 0 <= min(right):
    print("Yes")
    if left[-1] <= 0:
        ans = [LR[i][0] for i in range(N)]
        for i in range(N-1, -1, -1):
            if left[i+1] > 0:
                ans[i] += left[i+1]
                break
    else:
        ans = [LR[i][1] for i in range(N)]
        for i in range(N):
            if right[i+1] < 0:
                ans[i] += right[i+1]
                break
    print(*ans)
else:
    print("No")