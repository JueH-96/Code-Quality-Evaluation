def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    S = data[2]
    queries = data[3:]

    # Precompute prefix counts of '1' and '2'
    prefix_ones = [0] * (N + 1)
    prefix_twos = [0] * (N + 1)
    for i in range(N):
        prefix_ones[i + 1] = prefix_ones[i] + (1 if S[i] == '1' else 0)
        prefix_twos[i + 1] = prefix_twos[i] + (1 if S[i] == '2' else 0)

    # Collect positions of '/' in ascending order
    slash_positions = []
    for i in range(N):
        if S[i] == '/':
            slash_positions.append(i + 1)  # 1-based indexing

    # Helper to compute min(# of '1's before slash, # of '2's after slash) for a given slash
    def get_subseq_value(L, R, slash):
        # A = number of '1' in [L..slash-1]
        A = prefix_ones[slash - 1] - prefix_ones[L - 1]
        # B = number of '2' in [slash+1..R]
        B = prefix_twos[R] - prefix_twos[slash]
        # We take the minimum because the 11/22 string needs k '1's and k '2's
        return min(A, B)

    # For each query, find the slash that maximizes min(#1, #2)
    def answer_query(L, R):
        # Locate the subarray of slash_positions that lie in [L..R]
        left_index = bisect.bisect_left(slash_positions, L)
        right_index = bisect.bisect_right(slash_positions, R) - 1

        # If there is no slash at all in [L..R], answer is 0
        if left_index > right_index:
            return 0

        # We'll do a binary search to find the "crossing" index that balances A <= B
        # Because A grows with slash position, B shrinks with slash position,
        # min(A, B) is maximized around the crossing point of A and B.
        def check(mid):
            slash = slash_positions[mid]
            A = prefix_ones[slash - 1] - prefix_ones[L - 1]
            B = prefix_twos[R] - prefix_twos[slash]
            return A <= B

        l, r = left_index, right_index
        best = -1
        while l <= r:
            m = (l + r) // 2
            if check(m):
                best = m
                l = m + 1
            else:
                r = m - 1

        # Examine a handful of candidates around best plus boundary indices
        candidates = {best - 1, best, best + 1, left_index, right_index}
        ans_best = 0
        for c in candidates:
            if left_index <= c <= right_index and c != -1:
                slash = slash_positions[c]
                val = get_subseq_value(L, R, slash)
                if val > ans_best:
                    ans_best = val

        # If we found at least one slash, we can form a subsequence of length 2*k+1.
        # Here ans_best = k, so the final length is 2*k + 1.
        return 2 * ans_best + 1

        # If there's no slash, we already returned 0 above.

    # Process queries
    idx = 0
    out = []
    for _ in range(Q):
        Lq = int(queries[idx])
        Rq = int(queries[idx + 1])
        idx += 2
        out.append(str(answer_query(Lq, Rq)))

    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()