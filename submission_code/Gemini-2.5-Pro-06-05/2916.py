class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        """
        Determines if an array can be split into n single-element arrays based on a given rule.
        """
        n = len(nums)
        
        # An array with 1 or 2 elements can always be split down to single elements.
        # n=1: It's already a single element.
        # n=2: [a, b] -> [a], [b]. This split is always valid as resulting subarrays have length 1.
        if n <= 2:
            return True

        # dp[i][j] will be True if the subarray nums[i..j] can be split into single elements.
        dp = [[False] * n for _ in range(n)]

        # Precompute prefix sums for O(1) sum queries.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        def get_sum(i, j):
            return prefix_sum[j + 1] - prefix_sum[i]

        # Base case: All subarrays of length 1 are already split.
        for i in range(n):
            dp[i][i] = True

        # Iterate over subarray lengths, from 2 to n.
        for length in range(2, n + 1):
            # Iterate over all possible start indices for this length.
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Try all possible split points k for the subarray nums[i..j].
                # The split is into nums[i..k] and nums[k+1..j].
                for k in range(i, j):
                    # Check if the split itself is valid.
                    # A split of A into L and R is valid if:
                    # (len(L) == 1 or sum(L) >= m) AND (len(R) == 1 or sum(R) >= m)
                    
                    left_len = k - i + 1
                    right_len = j - k
                    
                    left_is_valid_piece = (left_len == 1) or (get_sum(i, k) >= m)
                    right_is_valid_piece = (right_len == 1) or (get_sum(k + 1, j) >= m)

                    if left_is_valid_piece and right_is_valid_piece:
                        # If the split is valid, we also need to ensure that the
                        # resulting subarrays can be further split down.
                        # This is where we use our DP results for smaller subproblems.
                        if dp[i][k] and dp[k+1][j]:
                            dp[i][j] = True
                            # We found a valid way to split nums[i..j], so we can break
                            # from the inner loop over k.
                            break
        
        return dp[0][n - 1]