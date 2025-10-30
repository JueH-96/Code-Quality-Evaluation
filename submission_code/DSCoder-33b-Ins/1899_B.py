# YOUR CODE HERE
import sys

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        a.sort()
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + a[i]
        ans = 0
        for k in range(1, n + 1):
            if n % k == 0:
                total = prefix[n]
                min_total = min(prefix[i] for i in range(k, n + 1, k))
                max_total = max(prefix[i] for i in range(k, n + 1, k))
                ans = max(ans, max_total - min_total)
        print(ans)

solve()