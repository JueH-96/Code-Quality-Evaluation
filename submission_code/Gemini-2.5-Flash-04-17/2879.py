import math

class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)

        # Precompute cost[i][j]: min changes to make s[i...j] a semi-palindrome
        # cost[i][j] will store the minimum changes for the substring s[i:j+1]
        # Indices i, j are 0-based start and end indices of the substring in the original string s.
        cost = [[float('inf')] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                sub_len = j - i + 1

                if sub_len < 2:
                    # Length 1 strings cannot be semi-palindromes by definition (1 <= d < len)
                    # cost remains infinity
                    continue

                min_changes_sub = float('inf')

                # Find all divisors d of sub_len, 1 <= d < sub_len
                divisors = [d for d in range(1, sub_len) if sub_len % d == 0]

                # If no such d exists (only happens if sub_len = 1, handled above),
                # the loop over divisors won't run and min_changes_sub remains inf.
                # If sub_len >= 2, there is always at least d=1 as a divisor < sub_len.

                for d in divisors:
                    current_changes = 0
                    
                    # For each starting remainder r modulo d
                    for r in range(d):
                        # The sequence of characters at indices i+r, i+r+d, i+r+2d, ... within s
                        # must form a palindrome.
                        seq_len = sub_len // d
                        
                        # Check if this sequence is a palindrome
                        # Compare first element with last, second with second-last, etc.
                        # The indices in the original string s are i + r + p1 * d
                        for p1 in range(seq_len // 2):
                            idx1_original = i + r + p1 * d
                            idx2_original = i + r + (seq_len - 1 - p1) * d
                            
                            if s[idx1_original] != s[idx2_original]:
                                current_changes += 1
                    min_changes_sub = min(min_changes_sub, current_changes)

                cost[i][j] = min_changes_sub

        # DP table dp[i][j]: min changes to partition s[0...i-1] into j semi-palindromes
        # i is the length of the prefix s[0...i-1], ranges from 0 to n
        # j is the number of partitions, ranges from 0 to k
        # dp[i][j] = min cost for s[:i] using j partitions
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]

        # Base case: 0 length prefix, 0 partitions, 0 cost
        dp[0][0] = 0

        # Fill DP table
        for j in range(1, k + 1): # Number of partitions (1 to k)
            # The total length of the prefix s[0...i-1] must be at least j*2 (each part >= 2)
            for i in range(j * 2, n + 1): # Current prefix s[0...i-1] (length i)
                
                # Iterate through all possible split points p
                # The split is after index p-1 in the original string s.
                # The first j-1 substrings partition s[0...p-1] (length p). The cost for this is dp[p][j-1].
                # The last substring is s[p...i-1] (length i-p). The cost for this is cost[p][i-1].

                # Constraints on p (the length of the prefix s[0...p-1] for j-1 partitions):
                # 1. The prefix s[0...p-1] (length p) must be partitionable into j-1 parts, each >= 2.
                #    Minimum length required is (j-1)*2. So p >= (j-1)*2.
                # 2. The last substring s[p...i-1] (length i-p) must have length >= 2.
                #    So i-p >= 2 => p <= i-2.

                # So p (the length of the prefix for j-1 partitions) ranges from (j-1)*2 up to i-2.
                # The loop range for p in Python is inclusive start, exclusive end.
                # We want p to go from (j-1)*2 up to i-2. So range is ((j-1)*2, i-1).
                for p in range((j - 1) * 2, i - 1): 
                    
                    # Check if the state dp[p][j-1] is reachable (not infinity)
                    if dp[p][j - 1] != float('inf'):
                        
                        # Get the cost for the last substring s[p...i-1].
                        # This corresponds to precomputed cost[p][i-1] where p is start index and i-1 is end index.
                        # The length is (i-1) - p + 1 = i - p.
                        # Since p <= i-2, i-p >= 2. The cost for this length >= 2 substring is always computed
                        # and stored in `cost`. It will be a finite integer.
                        current_sub_cost = cost[p][i - 1]
                        
                        # Update dp[i][j] with the minimum cost
                        dp[i][j] = min(dp[i][j], dp[p][j-1] + current_sub_cost)

        # The final answer is the minimum changes for the whole string s[0...n-1] (length n)
        # partitioned into k parts. This is dp[n][k].
        return dp[n][k]