# YOUR CODE HERE
def min_window_width(N, M, L):
    def can_fit(width):
        lines = 1
        current_width = 0
        for word_width in L:
            if current_width + word_width <= width:
                if current_width > 0:
                    current_width += 1  # Add space
                current_width += word_width
            else:
                lines += 1
                if lines > M:
                    return False
                current_width = word_width
        return True

    left = max(L)
    right = sum(L) + N - 1

    while left < right:
        mid = (left + right) // 2
        if can_fit(mid):
            right = mid
        else:
            left = mid + 1

    return left

N, M = map(int, input().split())
L = list(map(int, input().split()))

print(min_window_width(N, M, L))