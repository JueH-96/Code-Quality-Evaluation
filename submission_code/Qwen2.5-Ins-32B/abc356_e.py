# YOUR CODE HERE
import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    A.sort()
    count = [0] * (A[-1] + 1)
    for a in A:
        count[a] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    ans = 0
    for i in range(N):
        a = A[i]
        for k in range(1, 1001):
            l = a * k
            r = a * (k + 1) - 1
            if l > A[-1]:
                break
            ans += k * (count[min(r, A[-1])] - count[l - 1])
        ans -= a // a
    print(ans)

solve()