# YOUR CODE HERE
def can_fit_in_lines(words, max_width, max_lines):
    current_width = 0
    lines_used = 1
    
    for word in words:
        if current_width + word <= max_width:
            current_width += word + 1  # Add word and a space
        else:
            lines_used += 1
            current_width = word + 1  # Start new line with current word
            
        if lines_used > max_lines:
            return False
    
    return True

def find_minimum_width(N, M, lengths):
    left = max(lengths)  # Minimum width must be at least the width of the longest word
    right = sum(lengths) + (N - 1)  # Maximum width is all words in one line with spaces
    
    while left < right:
        mid = (left + right) // 2
        if can_fit_in_lines(lengths, mid, M):
            right = mid
        else:
            left = mid + 1
    
    return left

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
lengths = list(map(int, data[2:]))

print(find_minimum_width(N, M, lengths))