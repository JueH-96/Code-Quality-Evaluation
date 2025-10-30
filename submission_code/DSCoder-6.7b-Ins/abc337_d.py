import sys
from collections import deque

def solve():
    H, W, K = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    def check(S, H, W, K):
        for i in range(H):
            q = deque()
            for j in range(W):
                if S[i][j] == 'o':
                    q.append(j)
                    if len(q) > K:
                        q.popleft()
                else:
                    q = deque()
                if len(q) == K:
                    return True

        for j in range(W):
            q = deque()
            for i in range(H):
                if S[i][j] == 'o':
                    q.append(i)
                    if len(q) > K:
                        q.popleft()
                else:
                    q = deque()
                if len(q) == K:
                    return True

        return False

    def count_dots(S, H, W):
        return sum(row.count('.') for row in S)

    if check(S, H, W, K):
        print(0)
    elif count_dots(S, H, W) >= K:
        print(1)
    else:
        print(-1)

solve()