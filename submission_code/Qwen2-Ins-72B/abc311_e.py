import sys
from bisect import bisect_left, bisect_right

input = sys.stdin.readline

def main():
    H, W, N = map(int, input().split())
    AB = [list(map(int, input().split())) for _ in range(N)]

    AB.sort(key=lambda x: x[1])
    row = [[] for _ in range(H)]
    for a, b in AB:
        row[a-1].append(b-1)

    for i in range(H):
        row[i].append(W)
        row[i].insert(0, -1)
        for j in range(1, len(row[i])):
            row[i][j] -= row[i][j-1]

    AB.sort(key=lambda x: x[0])
    col = [[] for _ in range(W)]
    for a, b in AB:
        col[b-1].append(a-1)

    for i in range(W):
        col[i].append(H)
        col[i].insert(0, -1)
        for j in range(1, len(col[i])):
            col[i][j] -= col[i][j-1]

    ans = 0
    for i in range(H):
        for j in range(W):
            if not row[i] or not col[j]:
                continue
            l = 1
            r = min(len(row[i]), len(col[j])) + 1
            while r - l > 1:
                m = (l + r) // 2
                if bisect_right(row[i], row[i][bisect_left(row[i], m + j)] - j) - bisect_left(row[i], j) == bisect_right(col[j], col[j][bisect_left(col[j], m + i)] - i) - bisect_left(col[j], i):
                    l = m
                else:
                    r = m
            ans += l
    print(ans)