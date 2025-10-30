def can_fit_in_lines(words, max_width, max_lines):
    current_width = 0
    lines_used = 1
    
    for word in words:
        if current_width == 0:  # First word in a line
            current_width = word
        else:
            if current_width + 1 + word <= max_width:
                current_width += 1 + word
            else:
                lines_used += 1
                current_width = word
                if lines_used > max_lines:
                    return False
    return True

def find_minimum_width(N, M, word_lengths):
    left = max(word_lengths)  # Minimum width must be at least the longest word
    right = sum(word_lengths) + (N - 1)  # Maximum width if all in one line with spaces
    
    while left < right:
        mid = (left + right) // 2
        if can_fit_in_lines(word_lengths, mid, M):
            right = mid
        else:
            left = mid + 1
    
    return left

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
word_lengths = list(map(int, data[2:]))

result = find_minimum_width(N, M, word_lengths)
print(result)