import sys

def solve():
    n, k = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))

    if k == 4:
        count = sum(1 for x in a if x % 4 == 0)
        if count > 0:
            print(0)
            return
        count = sum(1 for x in a if x % 2 == 0)
        if count > 1:
            print(0)
            return
        if count == 1:
            print(1)
            return
        print(2)
        return

    ans = float('inf')
    for i in range(n):
        if a[i] % k == 0:
            print(0)
            return
        ans = min(ans, (k - (a[i] % k)) % k)

    if k == 2:
        print(ans)
        return

    count = sum(1 for x in a if x % 2 == 0)
    if count > 0:
        ans = min(ans, 1)

    print(ans)

t = int(sys.stdin.readline())
for _ in range(t):
    solve()