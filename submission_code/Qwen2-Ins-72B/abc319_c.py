from itertools import permutations
import sys
input = sys.stdin.readline

C = [list(map(int, input().split())) for _ in range(3)]

def check(p):
    for i in range(0, 9, 3):
        if p[i] == p[i + 1] and p[i] != p[i + 2]:
            return False
    for i in range(3):
        if p[i] == p[i + 3] and p[i] != p[i + 6]:
            return False
    if p[0] == p[4] and p[0] != p[8]:
        return False
    if p[2] == p[4] and p[2] != p[6]:
        return False
    return True

P = list(permutations([C[i][j] for i in range(3) for j in range(3)]))
cnt = 0
for p in P:
    if check(p):
        cnt += 1
print(cnt / len(P))