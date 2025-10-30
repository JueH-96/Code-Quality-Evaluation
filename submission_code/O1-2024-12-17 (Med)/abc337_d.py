def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, K = map(int, input_data[:3])
    S = input_data[3:]  # The grid, S[i] is the i-th row
    
    # A helper function that, given a "line" (string of length n containing '.', 'o', or 'x'),
    # returns the minimum number of flips ('.' -> 'o') needed to form a consecutive sub-segment
    # of length K with no 'x'. If no such sub-segment is possible, returns None.
    def min_cost_k_consecutive(line, K):
        n = len(line)
        ans = None
        i = 0
        while i < n:
            # skip 'x' to find the start of a possible segment
            if line[i] == 'x':
                i += 1
                continue
            # now line[i] is '.' or 'o', so let this be the start of a segment with no 'x'
            start = i
            while i < n and line[i] != 'x':
                i += 1
            end = i - 1  # segment is line[start..end], inclusive, and has no 'x'
            seg_len = end - start + 1
            if seg_len >= K:
                # Build a prefix sum array for '.' in this segment
                # seg: line[start:end+1]
                seg = line[start:end+1]
                dot = [0] * (seg_len + 1)  # dot[i] = number of '.' in seg[:i]
                for idx in range(seg_len):
                    dot[idx + 1] = dot[idx] + (1 if seg[idx] == '.' else 0)
                # Slide a window of size K over this segment
                for st in range(seg_len - K + 1):
                    cost = dot[st + K] - dot[st]  # number of '.' in seg[st:st+K]
                    if ans is None or cost < ans:
                        ans = cost
        return ans

    min_cost = None

    # Check all rows
    for r in range(H):
        res = min_cost_k_consecutive(S[r], K)
        if res is not None:
            if min_cost is None or res < min_cost:
                min_cost = res

    # Check all columns
    # To avoid storing all columns at once, we'll build each column as a string and process
    for c in range(W):
        col_line = []
        for r in range(H):
            col_line.append(S[r][c])
        col_line = ''.join(col_line)
        res = min_cost_k_consecutive(col_line, K)
        if res is not None:
            if min_cost is None or res < min_cost:
                min_cost = res

    # Print the result
    if min_cost is None:
        print(-1)
    else:
        print(min_cost)

# Do not forget to call main().
if __name__ == "__main__":
    main()