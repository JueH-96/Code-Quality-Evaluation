from collections import Counter
from sys import stdin, stdout

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    cnt = Counter(a)
    ans = 0
    for x in cnt:
        if x == 1 or x == 2:
            ans += cnt[x] * (n - 1)
            n -= cnt[x]
        else:
            ans += cnt[x] * (cnt[1] + cnt[2])
            n -= cnt[x]
    print(ans // 2)

t = int(input())
for _ in range(t):
    solve()