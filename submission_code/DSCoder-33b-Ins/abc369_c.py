# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    ans = 0
    for i in range(n):
        for j in range(i, n):
            if j == i:
                ans += 1
            else:
                if (a[j] - a[j - 1]) * (j - i) == a[j] - a[i] * (j - i):
                    ans += 1
    print(ans)

solve()