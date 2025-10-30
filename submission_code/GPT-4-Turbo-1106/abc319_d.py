def can_fit(words, max_width, max_lines):
    current_width = 0
    lines_used = 1
    for word in words:
        if current_width + word > max_width:
            lines_used += 1
            current_width = word + 1
        else:
            current_width += word + 1
    return lines_used <= max_lines

def find_min_window_width(N, M, word_lengths):
    left = max(word_lengths)  # Minimum possible width is the length of the longest word
    right = sum(word_lengths) + N  # Maximum possible width is the sum of all words plus spaces

    while left < right:
        mid = (left + right) // 2
        if can_fit(word_lengths, mid, M):
            right = mid
        else:
            left = mid + 1
    return left

# Read input
N, M = map(int, input().split())
word_lengths = list(map(int, input().split()))

# Find and print the minimum possible width of the window
print(find_min_window_width(N, M, word_lengths))