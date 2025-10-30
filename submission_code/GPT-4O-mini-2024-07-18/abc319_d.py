def can_fit_in_lines(words, max_width, max_lines):
    current_line_width = 0
    lines_used = 1  # Start with the first line

    for word in words:
        if current_line_width == 0:  # First word in the line
            current_line_width += word
        else:
            if current_line_width + 1 + word <= max_width:  # +1 for the space
                current_line_width += 1 + word
            else:
                lines_used += 1
                current_line_width = word  # Start a new line with the current word

                if lines_used > max_lines:  # Early exit if we exceed max lines
                    return False

    return lines_used <= max_lines

def minimum_window_width(N, M, lengths):
    left, right = max(lengths), sum(lengths) + (N - 1)  # max word length and total length with spaces
    answer = right

    while left <= right:
        mid = (left + right) // 2
        if can_fit_in_lines(lengths, mid, M):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1

    return answer

import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
M = int(data[1])
lengths = list(map(int, data[2:]))

result = minimum_window_width(N, M, lengths)
print(result)