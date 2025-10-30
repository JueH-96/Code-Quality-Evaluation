def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    words = [int(next(it)) for _ in range(N)]
    
    # Function to check if sentence fits into <= M lines given window width W
    def can_fit(W):
        lines = 1
        current_width = 0
        for length in words:
            if current_width == 0:
                # Start of a new line, put the word here.
                current_width = length
            elif current_width + 1 + length <= W:
                # Append the word on the same line with a gap of 1.
                current_width += 1 + length
            else:
                # Move to the next line for this word.
                lines += 1
                current_width = length
                if lines > M:
                    return False
        return lines <= M

    # Binary search for the minimum valid window width.
    # Lower bound is the maximum word length; upper bound is sum of all words plus spaces.
    low = max(words)
    high = sum(words) + (N - 1)
    
    while low < high:
        mid = (low + high) // 2
        if can_fit(mid):
            high = mid
        else:
            low = mid + 1
    print(low)

if __name__ == "__main__":
    main()