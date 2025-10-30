from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # n is the length of the array nums
        n = len(nums)
        # Create 2D lists dp and best of size n x n.
        # dp[i][j] will store the XOR score for the subarray nums[i..j].
        # best[i][j] will store the maximum XOR score among all subarrays contained in nums[i..j].
        dp = [[0] * n for _ in range(n)]
        best = [[0] * n for _ in range(n)]
        
        # For single-element subarrays, the XOR score is just the element itself.
        for i in range(n):
            dp[i][i] = nums[i]
            best[i][i] = nums[i]
        
        # Fill dp and best tables.
        # We use the recurrence: For any i < j:
        #     dp[i][j] = dp[i][j-1] XOR dp[i+1][j]
        # This recurrence can be verified by checking a couple of examples:
        # For a 2-element subarray [a, b]:
        #     dp[i][i+1] = a XOR b = dp[i][i] XOR dp[i+1][i+1].
        # For a 3-element subarray [a, b, c]:
        #     dp[i][i+2] = (a XOR b) XOR (b XOR c) = a XOR c.
        # Hence, the recurrence works.
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                dp[i][j] = dp[i][j - 1] ^ dp[i + 1][j]
                # best[i][j] is the maximum among:
                # - The XOR score for the full subarray nums[i..j] (that is dp[i][j])
                # - The best value for subarray nums[i+1..j]
                # - The best value for subarray nums[i..j-1]
                best[i][j] = max(dp[i][j], best[i + 1][j], best[i][j - 1])
        
        # Now, answer each query in O(1) time using the precomputed best[][] table.
        ans = []
        for l, r in queries:
            ans.append(best[l][r])
        return ans

# For quick testing (you can remove or comment out before submission)
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums = [2, 8, 4, 32, 16, 1]
    queries = [[0, 2], [1, 4], [0, 5]]
    print(sol.maximumSubarrayXor(nums, queries))  # Expected output: [12, 60, 60]
    
    # Example 2:
    nums = [0, 7, 3, 2, 8, 5, 1]
    queries = [[0, 3], [1, 5], [2, 4], [2, 6], [5, 6]]
    print(sol.maximumSubarrayXor(nums, queries))  # Expected output: [7, 14, 11, 14, 5]