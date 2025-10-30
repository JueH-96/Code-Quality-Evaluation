def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    L = [int(next(it)) for _ in range(N)]
    
    # The window width must at least be as wide as the longest word.
    lo = max(L)
    # In the worst case, all words are in one line with a space (width 1) between them.
    hi = sum(L) + (N - 1)
    
    # This function checks if it's possible to fit the sentence within M lines for a given window width.
    def can_fit(width):
        line_count = 1  # Start with the first line.
        current_width = 0
        for length in L:
            if current_width == 0:  
                # If we are at the beginning of a new line, just place the word.
                current_width = length
            else:
                # Otherwise, attempt to place the word with a preceding space.
                if current_width + 1 + length <= width:
                    current_width += 1 + length
                else:
                    # Place the word on a new line.
                    line_count += 1
                    current_width = length
                    # Early exit if we exceed the allowed number of lines.
                    if line_count > M:
                        return False
        return line_count <= M
    
    # Use binary search to find the minimal possible window width.
    ans = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_fit(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()