class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        """
        We want to place exactly 3 rooks on the chessboard so that no two rooks
        share the same row or column, and we want to maximize the sum of the
        board values where they are placed.

        A direct brute force over all (row, col) triples is O(m^3 * n^3), which is too large for m,n up to 100.

        Instead, we can exploit that we only need 3 rooks:

        1. Choose exactly 3 distinct columns c1, c2, c3 from the n available.
           There are C(n,3) such triplets.

        2. For each such triplet of columns, we consider each of the 6 permutations
           that assign one of those columns to "slot 0", one to "slot 1", and
           one to "slot 2".  (Because rooks don't share columns, we match each
           of the chosen columns to a different "rook" index.)

        3. For a fixed permutation p(0), p(1), p(2) of the three columns, define:
             A[r] = board[r][ c_{p(0)} ]
             B[r] = board[r][ c_{p(1)} ]
             C[r] = board[r][ c_{p(2)} ]
           We want to pick r1, r2, r3 all distinct to maximize A[r1] + B[r2] + C[r3].

           Notice that if we fix r1 and r2 (both distinct), we just need the best possible C[r3]
           for r3 != r1 and r3 != r2. We can find the top two values in C[], along with
           their indices, in O(m). Then for each (r1, r2), we do:
               partial_sum = A[r1] + B[r2]
               - if top1_index not in {r1, r2}, we add top1_value
               - else if top2_index not in {r1, r2}, we add top2_value
               - else skip (because the top 1 or 2 in C conflict)
           We'll keep track of the maximum over all (r1, r2).

        The inner loop over rows is O(m^2), and obtaining top1/top2 of C is O(m).
        Repeating for all 6 permutations is 6 * O(m^2) = O(m^2). So for each triple
        of columns, we spend O(m^2). There are C(n,3) = O(n^3) triples. With n,m up to 100,
        this is on the order of 6 * (n^3) * (m^2) ~ 6 * 10^10 in the worst case,
        which is quite large in Python but can sometimes be made to pass with an efficient
        implementation (depending on the judge/time-limit). In a lower-level language or
        with further optimizations, it is usually feasible.

        We'll implement it as described. This is the "3 columns + row matching" approach.
        """

        import sys
        input_data = sys.stdin.read()  # Not strictly needed if you call the function directly
        # In a coding platform you might parse input_data here.
        # For this solution method we assume board is already given.

        m = len(board)
        n = len(board[0]) if m > 0 else 0

        from itertools import combinations, permutations

        # Precompute board as a list of lists of ints (it might already be so).
        # board is given in that form, so we already have it.

        max_sum = -10**20  # something smaller than the minimum constraint

        # Helper to get top two elements of an array arr in O(m).
        # Returns (value1, index1, value2, index2) where value1 >= value2.
        def top_two(arr):
            top_val1, top_idx1 = -10**20, -1
            top_val2, top_idx2 = -10**20, -1
            for i, val in enumerate(arr):
                if val > top_val1:
                    top_val2, top_idx2 = top_val1, top_idx1
                    top_val1, top_idx1 = val, i
                elif val > top_val2:
                    top_val2, top_idx2 = val, i
            return top_val1, top_idx1, top_val2, top_idx2

        # Generate all combinations of 3 distinct columns
        for c1, c2, c3 in combinations(range(n), 3):
            # For each permutation of these columns, treat them as A/B/C
            for colA, colB, colC in permutations([c1, c2, c3], 3):
                # Build arrays A, B, C (length m)
                A = [board[r][colA] for r in range(m)]
                B = [board[r][colB] for r in range(m)]
                C_ = [board[r][colC] for r in range(m)]

                # Get top two from C_
                top1_val, top1_idx, top2_val, top2_idx = top_two(C_)

                # We'll do an O(m^2) pass over (r1, r2)
                # r1 != r2, but for simplicity we can just do a double loop and skip r1 == r2.
                local_max = -10**20

                for r1 in range(m):
                    valA = A[r1]
                    for r2 in range(m):
                        if r2 == r1:
                            continue
                        valB = B[r2]
                        # partial sum of A[r1] + B[r2]
                        part_sum = valA + valB

                        # We want the best possible C[r3] with r3 != r1 and r3 != r2
                        if top1_idx != r1 and top1_idx != r2:
                            # we can use top1
                            cand = part_sum + top1_val
                        elif top2_idx != r1 and top2_idx != r2:
                            # we must use top2
                            cand = part_sum + top2_val
                        else:
                            cand = -10**20  # can't place the third rook distinct from r1,r2

                        if cand > local_max:
                            local_max = cand

                # Update global max_sum
                if local_max > max_sum:
                    max_sum = local_max

        return max_sum