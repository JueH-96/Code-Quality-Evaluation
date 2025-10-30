from collections import deque

N, M = map(int, input().split())
LRS = [tuple(map(int, input().split())) for _ in range(N)]

blocked = [False] * (M + 2)
starts = deque()
ends = deque()
for L, R in LRS:
    blocked[L] = True
    starts.append(L)
    ends.append(R)

answer = 0
for m1 in range(M, 0, -1):
    if blocked[m1]:
        starts.pop()
        ends.pop()
    else:
        if starts and ends:
            l, r = starts[-1], ends[-1]
            l1 = max(l, m1)
            r1 = min(r, m1 + 1)
            answer += min(r1 - l1, m1 - m1 + 1)
        else:
            answer += m1 - m1 + 1

print(answer)