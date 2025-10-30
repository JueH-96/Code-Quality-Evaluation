def can_fit(words, W, M):
    """Check if words can fit in M lines with window width W"""
    lines = 1
    current_width = words[0]
    
    for i in range(1, len(words)):
        # Try to add word to current line (with space)
        if current_width + 1 + words[i] <= W:
            current_width += 1 + words[i]
        else:
            # Start a new line
            lines += 1
            current_width = words[i]
            if lines > M:
                return False
    
    return True

N, M = map(int, input().split())
L = list(map(int, input().split()))

# Binary search for minimum window width
left = max(L)  # At least the widest word
right = sum(L) + (N - 1)  # All words on one line

while left < right:
    mid = (left + right) // 2
    if can_fit(L, mid, M):
        right = mid
    else:
        left = mid + 1

print(left)