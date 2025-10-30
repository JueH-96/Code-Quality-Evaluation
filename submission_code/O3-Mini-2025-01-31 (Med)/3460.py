from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # We'll use a DP approach that computes for each prefix length i (i elements of the permutation)
        # the number of ways to achieve any inversion count <= 400.
        # A standard fact is that when you insert a new element into a permutation of length i,
        # the number of new inversions equals the position where you insert it (0 through i).
        #
        # Let dp[i][j] be the number of ways to arrange i elements with j inversions.
        # Transition:
        #   dp[i+1][j + p] += dp[i][j]   for p in range(0, i+1) provided j+p <= 400.
        #
        # Many requirements require a specific prefix (by end index, note that a requirement [end,cnt]
        # is on the prefix of length end+1). We enforce these conditions as soon as a prefix length is reached.
        
        # Build a mapping: key = prefix length (end + 1) must have inversion count = cnt.
        req_map = {}
        for end, cnt in requirements:
            req_map[end + 1] = cnt
        
        maxInv = 400  # We only care about inversion counts up to 400.
        
        # dp for 0 elements: dp[0] = 1 (zero inversions)
        dp = [0] * (maxInv + 1)
        dp[0] = 1
        
        # Process building permutation one element at a time.
        # i is the current number of placed elements; when inserting the next element,
        # it can be inserted in any of the i+1 positions resulting in 0..i additional inversions.
        for i in range(n):
            new_dp = [0] * (maxInv + 1)
            # Compute prefix sum of dp for fast range-sum queries.
            prefix = [0] * (maxInv + 1)
            prefix[0] = dp[0]
            for inv in range(1, maxInv + 1):
                prefix[inv] = (prefix[inv - 1] + dp[inv]) % MOD
            
            # For each possible total inversion count new_inv in the new dp array,
            # we want to add contributions dp[old_inv] for all possibilities of additional inversions p,
            # where new_inv = old_inv + p and p can be from 0 to i (or p <= new_inv).
            for new_inv in range(maxInv + 1):
                lo = new_inv - i if new_inv - i >= 0 else 0
                hi = new_inv  # old_inv must be ≤ new_inv
                # The sum over old_inv in [lo, hi] is computed quickly using prefix sums.
                if lo == 0:
                    new_dp[new_inv] = prefix[hi] % MOD
                else:
                    new_dp[new_inv] = (prefix[hi] - prefix[lo - 1]) % MOD
            
            # Enforce requirement if the new prefix length (i+1) has one.
            if (i + 1) in req_map:
                required = req_map[i + 1]
                temp = [0] * (maxInv + 1)
                if required <= maxInv:
                    temp[required] = new_dp[required] % MOD
                new_dp = temp
            
            dp = new_dp
        
        # By problem guarantee, there is at least one requirement with end == n-1,
        # meaning there is a condition for the full permutation (prefix length n).
        # The answer is forced to be the dp[n][req_map[n]].
        return dp[req_map[n]] % MOD

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # n = 3, requirements = [[2,2],[0,0]]
    # Permutations: [2, 0, 1] and [1, 2, 0]
    print(sol.numberOfPermutations(3, [[2, 2], [0, 0]]))  # Expected output: 2

    # Example 2:
    # n = 3, requirements = [[2,2],[1,1],[0,0]]
    # Only valid permutation is: [2, 0, 1]
    print(sol.numberOfPermutations(3, [[2, 2], [1, 1], [0, 0]]))  # Expected output: 1

    # Example 3:
    # n = 2, requirements = [[0,0],[1,0]]
    # Only valid permutation is: [0, 1]
    print(sol.numberOfPermutations(2, [[0, 0], [1, 0]]))  # Expected output: 1