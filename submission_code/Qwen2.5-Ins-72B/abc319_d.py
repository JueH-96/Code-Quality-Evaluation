# YOUR CODE HERE
import sys

def can_fit(words, max_width, max_lines):
    current_width = 0
    lines = 1
    for word in words:
        if current_width + word <= max_width:
            current_width += word + 1
        else:
            lines += 1
            current_width = word + 1
        if lines > max_lines:
            return False
    return True

def min_window_width(N, M, L):
    low, high = max(L), sum(L) + N - 1
    while low < high:
        mid = (low + high) // 2
        if can_fit(L, mid, M):
            high = mid
        else:
            low = mid + 1
    return low

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
L = list(map(int, data[2:]))
print(min_window_width(N, M, L))