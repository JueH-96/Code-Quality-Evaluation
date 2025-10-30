from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True

        # Compute prefix sums for quick range-sum calculations.
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Helper function: sum of nums[i:j+1]
        def sub_sum(i: int, j: int) -> int:
            return prefix[j+1] - prefix[i]
        
        # dp[i][j] will be True if the subarray nums[i:j+1] (which must have length >= 2)
        # can be split (via a sequence of valid splits) all the way into single-element arrays.
        #
        # Key observation: In any split, for each segment produced:
        #   - If its length is 1, it is automatically allowed.
        #   - If its length is 2, even if its sum is below m, it can be immediately split
        #     into two singletons (since the only split makes both parts length one).
        #   - For a segment with length >= 3, when produced from a split we require its sum >= m
        #     and that it is further splittable (if needed).
        
        dp = [[False] * n for _ in range(n)]
        
        # Base cases: a single element is a leaf.
        for i in range(n):
            dp[i][i] = True
        # For any segment of length 2, a valid (unique) split yields two singletons.
        for i in range(n-1):
            dp[i][i+1] = True
        
        # We now fill dp for segments of length L (>= 3)
        for L in range(3, n+1):
            for i in range(0, n - L + 1):
                j = i + L - 1
                # Try every possible valid split index k (splitting the segment into
                # left = nums[i:k+1] and right = nums[k+1:j+1])
                for k in range(i, j):
                    left_len = k - i + 1
                    right_len = j - k
                    
                    # Check left part:
                    # If it is a singleton or of length 2, no sum condition is needed.
                    # Otherwise, it must have sum at least m and be splittable.
                    if left_len == 1 or left_len == 2:
                        left_valid = True
                    else:
                        left_valid = (sub_sum(i, k) >= m) and dp[i][k]
                    
                    # Check right part similarly.
                    if right_len == 1 or right_len == 2:
                        right_valid = True
                    else:
                        right_valid = (sub_sum(k+1, j) >= m) and dp[k+1][j]
                    
                    if left_valid and right_valid:
                        dp[i][j] = True
                        break
        
        # Finally, we want to know if the entire array (nums[0:n]) can be split
        # into singletons following the rule.
        return dp[0][n-1]

# Below are some simple tests.
if __name__ == "__main__":
    sol = Solution()
    # Example 1: Expected True
    print(sol.canSplitArray([2, 2, 1], 4))
    # Example 2: Expected False
    print(sol.canSplitArray([2, 1, 3], 5))
    # Example 3: Expected True
    print(sol.canSplitArray([2, 3, 3, 2, 3], 6))