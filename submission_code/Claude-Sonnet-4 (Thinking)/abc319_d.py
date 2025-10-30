def can_fit(words, W, M):
    lines = 1
    current_width = 0
    
    for i, word_width in enumerate(words):
        if i == 0:
            current_width = word_width
        else:
            # Try to add this word to current line
            if current_width + 1 + word_width <= W:
                current_width += 1 + word_width
            else:
                # Start new line
                lines += 1
                current_width = word_width
                if lines > M:
                    return False
    
    return True

N, M = map(int, input().split())
L = list(map(int, input().split()))

# Binary search on the answer
left = max(L)  # minimum possible width
right = sum(L) + (N - 1)  # maximum possible width

while left < right:
    mid = (left + right) // 2
    if can_fit(L, mid, M):
        right = mid
    else:
        left = mid + 1

print(left)