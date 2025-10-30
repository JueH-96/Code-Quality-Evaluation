class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        dp = {}
        max_length = 1
        
        for num in nums:
            # Check if num can be part of a sequence ending at num
            if num - 1 in dp:
                dp[num] = dp[num - 1] + 1
            else:
                dp[num] = 1
            # Check if num + 1 can be part of a sequence ending at num + 1
            if num + 1 in dp:
                dp[num + 1] = dp[num] + 1
            else:
                dp[num + 1] = dp[num] if num in dp else 1
            # Update the maximum length
            max_length = max(max_length, dp[num], dp.get(num + 1, 0))
        
        return max_length