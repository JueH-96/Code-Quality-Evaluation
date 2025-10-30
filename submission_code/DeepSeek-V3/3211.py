class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Initialize dp array
        dp = [1] * n
        # Initialize prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        # To keep track of the earliest j where prefix[j] <= prefix[i] - nums[i]
        # We can use a list to store the prefix sums and perform binary search
        # But for simplicity, we will use a linear search here
        # For better performance, a binary search approach can be used
        for i in range(1, n):
            # Find the earliest j where prefix[j] <= prefix[i+1] - nums[i]
            # Since prefix is non-decreasing, we can find the last j where prefix[j] <= prefix[i+1] - nums[i]
            # We can use binary search for this
            target = prefix[i+1] - nums[i]
            low = 0
            high = i
            best_j = -1
            while low <= high:
                mid = (low + high) // 2
                if prefix[mid] <= target:
                    best_j = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if best_j != -1:
                dp[i] = dp[best_j-1] + 1
            else:
                dp[i] = 1
        return max(dp)