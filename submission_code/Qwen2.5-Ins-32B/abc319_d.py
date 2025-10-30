import sys

def min_window_width(N, M, L):
    def can_fit(W):
        lines, current_width = 1, 0
        for word in L:
            if current_width + word + (1 if current_width > 0 else 0) <= W:
                current_width += word + (1 if current_width > 0 else 0)
            else:
                lines += 1
                current_width = word
        return lines <= M

    left, right = max(L), sum(L) + len(L) - 1
    while left < right:
        mid = (left + right) // 2
        if can_fit(mid):
            right = mid
        else:
            left = mid + 1
    return left

input = sys.stdin.read
data = input().split()
N, M = map(int, data[:2])
L = list(map(int, data[2:]))
print(min_window_width(N, M, L))