# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))

    def check(mid):
        cnt = 1
        sum = 0
        for i in range(N):
            if sum + L[i] > mid:
                cnt += 1
                sum = L[i]
            else:
                sum += L[i]
        return cnt <= M

    left = max(L)
    right = sum(L)
    while left < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid + 1
    print(left)

solve()