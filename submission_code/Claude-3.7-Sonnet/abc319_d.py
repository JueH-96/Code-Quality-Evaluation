def can_fit(words, max_lines, window_width):
    """Check if the words can fit within max_lines lines with the given window_width."""
    if max(words) > window_width:  # Ensure each word can fit within the window
        return False
    
    lines_used = 1
    current_line_width = words[0]
    
    for i in range(1, len(words)):
        # Try to add the word to the current line (with a space)
        if current_line_width + 1 + words[i] <= window_width:
            current_line_width += 1 + words[i]
        else:
            # Start a new line
            lines_used += 1
            current_line_width = words[i]
    
    return lines_used <= max_lines

def min_window_width(words, max_lines):
    """Find the minimum window width where the sentence can fit within max_lines lines."""
    # Define the search range
    left = max(words)  # Minimum possible width is the width of the widest word
    right = sum(words) + len(words) - 1  # Maximum possible width (all on one line)
    
    while left < right:
        mid = (left + right) // 2
        if can_fit(words, max_lines, mid):
            right = mid
        else:
            left = mid + 1
    
    return left

def main():
    N, M = map(int, input().split())
    L = list(map(int, input().split()))
    
    result = min_window_width(L, M)
    print(result)

if __name__ == "__main__":
    main()