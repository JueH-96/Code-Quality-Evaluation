from typing import List

MOD = 10**9 + 7

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i][v] will represent the count of ways to pick arr1[0..i]
        # such that arr1[i] == v.
        # Note: at each index i, valid v are 0 <= v <= nums[i] (since arr1[i]+arr2[i]=nums[i]).
        dp = [[0] * 51 for _ in range(n)]
        
        # Base: for index 0, arr1[0] can be any value from 0 to nums[0]
        for v in range(nums[0] + 1):
            dp[0][v] = 1
        
        # Transition:
        # For each consecutive index, let i-1 have arr1[i-1] = v.
        # At index i, we choose arr1[i] = y such that:
        #  1. y is in [0, nums[i]] (from the sum condition).
        #  2. arr1 is non-decreasing: y >= v.
        #  3. arr2 is non-increasing.
        #     Since arr2[i-1] = nums[i-1] - v, and arr2[i] = nums[i] - y, we need:
        #         nums[i] - y <= nums[i-1] - v   =>   y >= v + (nums[i] - nums[i-1]).
        #
        # So the necessary lower bound at index i is:
        #     lower = v + max(0, nums[i] - nums[i-1])
        # and we must have y in [lower, nums[i]].
        for i in range(1, n):
            # Calculate the extra lower requirement coming from monotonicity of arr2:
            diff = nums[i] - nums[i-1]
            add_val = diff if diff > 0 else 0
            # Transition from index i-1 (with value v) to index i (with value y)
            for v in range(nums[i-1] + 1):
                if dp[i-1][v] == 0:
                    continue
                lower = v + add_val
                if lower > nums[i]:
                    continue
                for y in range(lower, nums[i] + 1):
                    dp[i][y] = (dp[i][y] + dp[i-1][v]) % MOD
                    
        # The answer is the sum of ways for the last element:
        return sum(dp[n-1][:nums[n-1] + 1]) % MOD

# For quick testing:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countOfPairs([2, 3, 2]))    # Expected output: 4
    print(sol.countOfPairs([5, 5, 5, 5])) # Expected output: 126