def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    words = [int(next(it)) for _ in range(N)]
    
    # Function to check if the sentence can be displayed in at most M lines with window width w.
    def feasible(w):
        lines = 1
        current_width = words[0]
        for i in range(1, N):
            if current_width + 1 + words[i] <= w:
                current_width += 1 + words[i]
            else:
                lines += 1
                current_width = words[i]
                if lines > M:
                    return False
        return True

    # Lower bound: at least the maximum word width.
    # Upper bound: the sum of all word widths plus (N-1) spaces.
    low = max(words)
    high = sum(words) + (N - 1)
    
    # Binary search for the minimum width that is feasible.
    while low < high:
        mid = (low + high) // 2
        if feasible(mid):
            high = mid
        else:
            low = mid + 1

    sys.stdout.write(str(low))

if __name__ == '__main__':
    main()