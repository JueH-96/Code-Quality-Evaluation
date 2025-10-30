import sys

def min_window_width(N, M, L):
    left, right = 0, 10**18
    while right - left > 1:
        mid = (left + right) // 2
        lines = 1
        width = 0
        for i in range(N):
            if width + L[i] > mid:
                lines += 1
                width = L[i]
            else:
                width += L[i] + 1
        if lines > M:
            left = mid
        else:
            right = mid
    return right

N, M = map(int, sys.stdin.readline().split())
L = list(map(int, sys.stdin.readline().split()))
print(min_window_width(N, M, L))