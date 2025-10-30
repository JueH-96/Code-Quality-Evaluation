def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    L = list(map(int, data[2:]))

    # The minimum possible width cannot be smaller than the largest word:
    lo = max(L)
    # The maximum possible width is placing all words on one line:
    hi = sum(L) + (N - 1)  # (N-1) spaces

    def can_fit(width):
        """Check if it's possible to fit all words within M lines of given width."""
        lines_used = 1
        current_line_width = L[0]

        for i in range(1, N):
            # If next word fits on the current line (with the space),
            # place it there. Otherwise, start a new line.
            if current_line_width + 1 + L[i] <= width:
                current_line_width += 1 + L[i]
            else:
                lines_used += 1
                current_line_width = L[i]
                # If we've used more than M lines, it's not feasible
                if lines_used > M:
                    return False
        return True

    # Binary search to find the minimum feasible window width
    while lo < hi:
        mid = (lo + hi) // 2
        if can_fit(mid):
            hi = mid
        else:
            lo = mid + 1

    print(lo)

# Do not forget to call main function
if __name__ == "__main__":
    main()