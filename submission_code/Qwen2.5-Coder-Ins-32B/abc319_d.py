# YOUR CODE HERE
def can_fit_in_lines(width, L, M):
    lines = 1
    current_width = 0
    for i in range(len(L)):
        if current_width + L[i] > width:
            lines += 1
            current_width = L[i]
            if lines > M:
                return False
        else:
            current_width += L[i] + 1 if i > 0 else L[i]
    return True

def min_window_width(N, M, L):
    left = max(L)
    right = sum(L) + N - 1
    while left < right:
        mid = (left + right) // 2
        if can_fit_in_lines(mid, L, M):
            right = mid
        else:
            left = mid + 1
    return left

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
L = list(map(int, input[2:]))
print(min_window_width(N, M, L))