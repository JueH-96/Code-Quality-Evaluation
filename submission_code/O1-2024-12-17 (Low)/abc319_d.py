def can_fit(width, M, lengths):
    """
    Check if it's possible to fit all words into M lines with the given window width.
    lengths: list of word widths
    """
    lines_used = 1  # start with the first line
    current_line_width = lengths[0]  # place the first word
    
    for i in range(1, len(lengths)):
        # Check if we can append the i-th word to the current line
        if current_line_width + 1 + lengths[i] <= width:
            # Append the word to the current line
            current_line_width += 1 + lengths[i]
        else:
            # Move to the next line
            lines_used += 1
            current_line_width = lengths[i]
            # If we exceed the allowed number of lines, no need to continue
            if lines_used > M:
                return False
    return True

def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    lengths = list(map(int, data[2:]))

    left = max(lengths)  # Minimum possible width can't be smaller than the largest word
    right = sum(lengths) + (N - 1)  # Maximum possible width if all words are in a single line with spaces

    # Binary search to find the minimal width that fits in M lines
    ans = right
    while left <= right:
        mid = (left + right) // 2
        if can_fit(mid, M, lengths):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1

    print(ans)

# Call main to run the solution
if __name__ == "__main__":
    main()