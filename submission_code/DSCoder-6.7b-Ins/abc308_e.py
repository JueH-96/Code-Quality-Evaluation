import sys

def mex(a, b, c):
    s = set([a, b, c])
    i = 0
    while True:
        if i not in s:
            return i
        i += 1

def solve(n, a, s):
    prefix_mex = [0] * (n + 1)
    prefix_count = [[0, 0, 0] for _ in range(n + 1)]
    for i in range(1, n + 1):
        prefix_count[i][a[i - 1]] += 1
        for j in range(3):
            prefix_count[i][j] += prefix_count[i - 1][j]
        prefix_mex[i] = prefix_mex[i - 1] + mex(*prefix_count[i])
    total = 0
    for i in range(n):
        if s[i] == 'M':
            total += prefix_mex[i]
            total += mex(*prefix_count[i + 1])
        elif s[i] == 'E':
            total += mex(*prefix_count[i])
            total += mex(*prefix_count[i + 1])
        else:
            total += mex(*prefix_count[i + 1])
    return total

n = int(sys.stdin.readline().strip())
a = list(map(int, sys.stdin.readline().strip().split()))
s = sys.stdin.readline().strip()
print(solve(n, a, s))