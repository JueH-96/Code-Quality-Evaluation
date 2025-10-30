import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    def check(k):
        left, right = 0, N - 1
        while left <= right:
            if A[left] > k:
                left += 1
            elif A[right] > k:
                right -= 1
            elif A[left] == k and A[right] == k:
                left += 1
                right -= 1
                k -= 1
            else:
                return False
        return True

    lo, hi = 1, N + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if check(mid):
            lo = mid
        else:
            hi = mid - 1

    print(lo)

if __name__ == "__main__":
    solve()