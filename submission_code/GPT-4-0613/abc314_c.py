from collections import defaultdict
from sys import stdin, stdout

def solve():
    n, m = map(int, stdin.readline().split())
    s = list(stdin.readline().strip())
    c = list(map(int, stdin.readline().split()))
    color = defaultdict(list)
    for i in range(n):
        color[c[i]].append(s[i])
    for i in range(1, m + 1):
        color[i] = color[i][::-1]
    for i in range(n):
        s[i] = color[c[i]].pop()
    stdout.write(''.join(s))

solve()