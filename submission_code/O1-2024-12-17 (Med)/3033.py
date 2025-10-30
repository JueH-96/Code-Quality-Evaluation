class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        """
        We want to transform s1 into s2 by flipping bits in pairs:
          1) Flip any two bits (cost = x).
          2) Flip two adjacent bits (cost = 1).
        Return the minimum cost or -1 if impossible.

        Key observations/approach:
         • Each flip inverts exactly two bits in s1.
         • Therefore, the total number of differing bits (the Hamming distance)
           between s1 and s2 must be even; otherwise it's impossible.
         • We will simulate fixing mismatches from left to right.
         • Whenever s1[i] != s2[i], we look ahead:
              - If there is also a mismatch at i+1, sometimes we can fix both
                together cheaply with the adjacent-flip operation (cost 1),
                or (in the "cross" case) we can fix them with two adjacent flips
                (cost 2) vs one "any-two-bits" flip (cost x).  We choose the cheaper.
              - Otherwise, we pair the mismatch at i with some future mismatch j
                using the cost=x operation.
         • If at any point we cannot find another mismatch to pair with, the task
           is impossible (-1).

        This greedy strategy covers all optimal cases:
         - Adjacent mismatches that can be immediately fixed via a single cost=1 flip
           should be done so when they are the same bit (e.g. both '0' in s1 vs '1' in s2).
         - Two "swapped" mismatches (cross-mismatch: '0' vs '1' and '1' vs '0') can
           be fixed by two adjacent flips for cost=2 or a single "any two bits" flip
           for cost=x; we pick min(2, x).
         - Everything else is handled by pairing mismatches with cost=x.
        """

        n = len(s1)
        # Quick impossibility check: the number of differing bits must be even.
        diff_bits = sum(a != b for a, b in zip(s1, s2))
        if diff_bits % 2 != 0:
            return -1

        # Work with mutable lists of bits (0/1) instead of strings for easier flipping.
        A = [int(ch) for ch in s1]
        B = [int(ch) for ch in s2]

        cost = 0
        i = 0
        while i < n:
            if A[i] == B[i]:
                i += 1
                continue

            # We have a mismatch at i.
            # Check if there's also a mismatch immediately at i+1.
            if i + 1 < n and A[i+1] != B[i+1]:
                # We have two adjacent mismatches at i and i+1.

                # Case 1: "direct mismatch" => s1[i] == s1[i+1], s2[i] == s2[i+1] (but they differ from each other).
                #   e.g. A[i],A[i+1] = 0,0 and B[i],B[i+1] = 1,1.
                #   A single adjacent flip (cost=1) will fix both bits (00->11 or 11->00).
                #   Or we flip them with the cost-x operation. We choose min(1, x).
                if A[i] == A[i+1] and B[i] == B[i+1]:
                    cost += min(x, 1)
                    # Flip them in A to match B:
                    A[i]   = B[i]
                    A[i+1] = B[i+1]
                    i += 2
                    continue

                # Case 2: "cross mismatch" => A[i] != A[i+1], B[i] != B[i+1].
                #   Typically this means something like A[i],A[i+1] = (0,1), B[i],B[i+1] = (1,0),
                #   so we could fix them by two adjacent flips (cost=2) or one "any-two-bits" flip
                #   (cost=x). We pick min(2, x).
                #   NOTE: The actual bits should be complementary in a "cross" sense.  In practice,
                #         if A[i],A[i+1] differ and B[i],B[i+1] differ, the mismatches can be
                #         resolved by 2 adjacency-flips or 1 cost-x flip.
                if A[i] != A[i+1] and B[i] != B[i+1]:
                    # Cost is min(2, x).
                    cost += min(2, x)
                    # Fix them to match B
                    A[i]   = B[i]
                    A[i+1] = B[i+1]
                    i += 2
                    continue

                # Otherwise, pair these two mismatches with cost=x and fix them.
                cost += x
                A[i]   = B[i]
                A[i+1] = B[i+1]
                i += 2

            else:
                # Only one mismatch visible at i; we must pair it with some future mismatch j.
                cost += x

                # Find a future mismatch j > i.
                j = i + 1
                while j < n and A[j] == B[j]:
                    j += 1
                if j >= n:
                    # No partner mismatch found -> impossible
                    return -1

                # Flip both bits i and j to match B.
                A[i] = B[i]
                A[j] = B[j]
                i += 1  # move on

        # Final check
        return cost if A == B else -1