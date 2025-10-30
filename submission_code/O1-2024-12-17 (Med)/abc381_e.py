def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    S = input_data[2]
    queries = input_data[3:]
    
    # For convenience, use 1-based indexing for S.
    # So let's pad S with a dummy character at index 0.
    # Then S[i] for i in [1..N].
    S = " " + S  # Now S[1] is the first real character.
    
    # Precompute prefix counts of '1' and '2'.
    prefix_1 = [0] * (N + 1)  # prefix_1[i] = number of '1's in S[1..i]
    prefix_2 = [0] * (N + 1)  # prefix_2[i] = number of '2's in S[1..i]
    for i in range(1, N + 1):
        prefix_1[i] = prefix_1[i - 1] + (1 if S[i] == '1' else 0)
        prefix_2[i] = prefix_2[i - 1] + (1 if S[i] == '2' else 0)
    
    # Gather all slash positions and their "val = prefix_1[j-1] + prefix_2[j]".
    slash_pos = []
    slash_val = []
    for j in range(1, N + 1):
        if S[j] == '/':
            slash_pos.append(j)
            slash_val.append(prefix_1[j - 1] + prefix_2[j])
    
    # We'll define a helper to answer a single query (L, R).
    # We want to find the maximum of min(#1_left, #2_right) among slashes j in [L..R].
    # Where #1_left = prefix_1[j-1] - prefix_1[L-1], #2_right = prefix_2[R] - prefix_2[j].
    # Then answer = 2 * that_max + 1 (if there is at least one slash), else 0.
    
    def answer_query(L, R):
        # 1) Find the subrange of slash_pos that lies in [L..R].
        pL = bisect.bisect_left(slash_pos, L)
        pR = bisect.bisect_right(slash_pos, R) - 1
        if pL > pR:
            # No slash in [L..R].
            return 0
        
        # 2) Define T = prefix_1[L-1] + prefix_2[R].
        T = prefix_1[L - 1] + prefix_2[R]
        
        # 3) In slash_val[pL..pR], find the index close to T.
        #    We'll use bisect_right to find the largest index i where slash_val[i] <= T.
        idx = bisect.bisect_right(slash_val, T, pL, pR + 1) - 1
        
        # We'll check up to a few neighbors around idx to find the best min(...).
        candidates = [idx - 1, idx, idx + 1]
        
        best_f = 0  # best min(#1_left, #2_right) found so far
        
        for c in candidates:
            if pL <= c <= pR:
                j = slash_pos[c]
                # Compute the number of 1s to the left (within [L..j-1]) and 2s to the right (within [j+1..R]).
                left_ones = prefix_1[j - 1] - prefix_1[L - 1]
                right_twos = prefix_2[R] - prefix_2[j]
                f = min(left_ones, right_twos)
                if f > best_f:
                    best_f = f
        
        # If there was at least one slash, we can form at least length=1 (the slash alone).
        # The formula for the 11/22 subsequence length is 2*best_f + 1.
        # best_f >= 0, so this is valid. If no slash, we already returned 0.
        return 2 * best_f + 1
    
    # Process each query.
    idx = 0
    out = []
    for _ in range(Q):
        L = int(queries[idx]); R = int(queries[idx + 1])
        idx += 2
        out.append(str(answer_query(L, R)))
    
    # Print all answers.
    print("
".join(out))

# Do not forget to call main().
if __name__ == "__main__":
    main()