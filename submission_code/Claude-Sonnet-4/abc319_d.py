def can_fit(words, width, max_lines):
    lines = 1
    current_width = 0
    
    for i, word_len in enumerate(words):
        if i == 0:
            # First word goes to first line
            current_width = word_len
        else:
            # Try to add word to current line (with space)
            needed_width = current_width + 1 + word_len
            if needed_width <= width:
                current_width = needed_width
            else:
                # Start new line
                lines += 1
                if lines > max_lines:
                    return False
                current_width = word_len
    
    return True

# Read input
n, m = map(int, input().split())
words = list(map(int, input().split()))

# Binary search for minimum width
left = max(words)  # At least the width of the longest word
right = sum(words) + n - 1  # All words on one line with spaces

while left < right:
    mid = (left + right) // 2
    if can_fit(words, mid, m):
        right = mid
    else:
        left = mid + 1

print(left)