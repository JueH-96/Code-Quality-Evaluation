import sys
from bisect import bisect_left

def solve():
    input = sys.stdin.read
    n, *a = map(int, input().split())
    a.reverse()
    bit = [0] * (n + 1)
    def add(i, x):
        i += 1
        while i <= n:
            bit[i] += x
            i += i & -i
    def sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s
    ans = 0
    for i, x in enumerate(a):
        ans += x * (i - sum(bisect_left(a, x)))
        add(bisect_left(a, x), 1)
    print(ans)

solve()