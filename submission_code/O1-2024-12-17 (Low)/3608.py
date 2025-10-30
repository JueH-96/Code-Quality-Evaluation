class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        """
        We want the number of ways to pick two disjoint non-empty subsequences of nums
        whose GCDs are the same.  We can solve this by dynamic programming over
        (index, gcd_of_subseq1, gcd_of_subseq2, emptiness_mask), where emptiness_mask
        tracks which of the two subsequences have become non-empty.

        Let dp[g1][g2][mask] = number of ways (mod 1e9+7) to form two subsequences
        from the elements processed so far with gcds = g1, g2, and mask describing
        emptiness (bit0=1 if subseq1 is non-empty, bit1=1 if subseq2 is non-empty).

        We iterate over each element x.  For each state (g1, g2, mask), we can:
          - Skip x  (leave dp state unchanged)
          - Add x to subsequence1  => new_g1 = gcd(g1, x) if already non-empty, else x
          - Add x to subsequence2  => similarly.

        At the end, we sum dp[g][g][3] for g=1..200 (mask=3 means both subsequences non-empty).

        The complexity is O(n * 201 * 201 * 4) which is acceptable for n <= 200 if done carefully.
        """

        import math
        MOD = 10**9 + 7

        # Precompute gcd for all pairs up to 200 to speed up transitions
        max_val = 200
        gcd_table = [[0]*(max_val+1) for _ in range(max_val+1)]
        for a in range(max_val+1):
            for b in range(max_val+1):
                if a == 0:
                    gcd_table[a][b] = b
                elif b == 0:
                    gcd_table[a][b] = a
                else:
                    gcd_table[a][b] = math.gcd(a, b)

        # dp[g1][g2][mask] after processing some prefix of nums
        # We'll use a rolling array approach: dp_curr, dp_next
        dp_curr = [[[0]*4 for _ in range(max_val+1)] for _ in range(max_val+1)]
        dp_curr[0][0][0] = 1  # initially, both subsequences are empty (gcd=0 means empty)

        for x in nums:
            dp_next = [[[0]*4 for _ in range(max_val+1)] for _ in range(max_val+1)]
            for g1 in range(max_val+1):
                for g2 in range(max_val+1):
                    for mask in range(4):
                        count = dp_curr[g1][g2][mask]
                        if count == 0:
                            continue

                        # 1) Skip this element
                        dp_next[g1][g2][mask] = (dp_next[g1][g2][mask] + count) % MOD

                        # 2) Add this element to subsequence1
                        # If subsequence1 is empty (mask bit 0 not set), gcd becomes x
                        # else gcd becomes gcd(g1, x).
                        new_g1 = x if (mask & 1) == 0 else gcd_table[g1][x]
                        new_mask = mask | 1  # mark subsequence1 as non-empty
                        dp_next[new_g1][g2][new_mask] = (
                            dp_next[new_g1][g2][new_mask] + count
                        ) % MOD

                        # 3) Add this element to subsequence2
                        new_g2 = x if (mask & 2) == 0 else gcd_table[g2][x]
                        new_mask = mask | 2  # mark subsequence2 as non-empty
                        dp_next[g1][new_g2][new_mask] = (
                            dp_next[g1][new_g2][new_mask] + count
                        ) % MOD

            dp_curr = dp_next

        # Sum up dp[g][g][3] for g>=1 to get the final count
        ans = 0
        for g in range(1, max_val+1):
            ans = (ans + dp_curr[g][g][3]) % MOD

        return ans