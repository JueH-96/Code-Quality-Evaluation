def can_fit(N, M, L, W):
    lines = 1
    current_width = 0
    for i in range(N):
        if current_width == 0:
            current_width = L[i]
        else:
            if current_width + 1 + L[i] <= W:
                current_width += 1 + L[i]
            else:
                lines += 1
                current_width = L[i]
        if lines > M:
            return False
    return True

def find_min_width(N, M, L):
    left = max(L)
    right = sum(L) + (N - 1)
    while left < right:
        mid = (left + right) // 2
        if can_fit(N, M, L, mid):
            right = mid
        else:
            left = mid + 1
    return left

# Read input
N, M = map(int, input().split())
L = list(map(int, input().split()))

# Find and print the minimum width
print(find_min_width(N, M, L))