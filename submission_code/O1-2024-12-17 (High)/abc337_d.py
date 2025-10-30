def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, K = map(int, data[:3])
    grid = data[3:]  # list of H strings

    # A helper function.
    # Given a "line" (string or list of chars) and integer K,
    # it returns the minimum number of '.' we must convert to 'o'
    # to obtain at least one contiguous block of length K (without 'x') with all 'o'.
    # If impossible in this line, return None.
    def min_cost_in_line(line, K):
        n = len(line)
        best_in_line = None
        i = 0
        while i < n:
            # skip over 'x' to find a consecutive segment without 'x'
            if line[i] == 'x':
                i += 1
                continue
            start = i
            # find the maximal sub-segment without 'x'
            while i < n and line[i] != 'x':
                i += 1
            seg_len = i - start
            # If this segment has length >= K, we can try sliding windows of size K
            if seg_len >= K:
                segment = line[start:start + seg_len]
                # Build prefix sums of '.' counts
                prefix = [0] * (seg_len + 1)
                for idx in range(seg_len):
                    prefix[idx + 1] = prefix[idx] + (1 if segment[idx] == '.' else 0)
                # Now check all K-length windows
                for left in range(seg_len - K + 1):
                    cost = prefix[left + K] - prefix[left]  # number of '.' in this window
                    if best_in_line is None or cost < best_in_line:
                        best_in_line = cost
                        if best_in_line == 0:
                            return 0  # cannot do better than 0
        return best_in_line

    best = None

    # 1) Check rows
    for i in range(H):
        cost = min_cost_in_line(grid[i], K)
        if cost is not None:
            if best is None or cost < best:
                best = cost
            if best == 0:  # cannot do better than 0
                break

    # 2) Check columns (only if best != 0 so far)
    if best != 0:  
        for j in range(W):
            # Build the j-th column as a string
            col = ''.join(grid[i][j] for i in range(H))
            cost = min_cost_in_line(col, K)
            if cost is not None:
                if best is None or cost < best:
                    best = cost
                if best == 0:
                    break

    # 3) Output result
    if best is None:
        print(-1)
    else:
        print(best)

# Don't forget to call main()
if __name__ == "__main__":
    main()