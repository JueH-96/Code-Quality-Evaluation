class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # If the total sum is odd, there can't be any even-sum partitions
        if total_sum % 2 != 0:
            return 0
        
        target_sum = total_sum // 2
        
        # Create a 2D dynamic programming table
        dp = [[False] * (target_sum + 1) for _ in range(n + 1)]
        
        # Base case: with 0 elements, we can only achieve a sum of 0
        for i in range(target_sum + 1):
            dp[0][i] = False
        
        # Base case: with 1 element, we can achieve any sum up to that element
        for i in range(1, n + 1):
            dp[i][0] = True
        
        # Fill the dynamic programming table
        for i in range(1, n + 1):
            for j in range(1, target_sum + 1):
                if nums[i - 1] > j:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
        
        # Count the number of partitions with even sum difference
        count = 0
        for j in range(target_sum + 1):
            if dp[n][j]:
                count += 1
        
        return count