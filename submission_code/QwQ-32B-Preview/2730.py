class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        dp = [0] * (k + 1)
        
        for num in nums:
            for operations in range(k, -1, -1):
                current_or = dp[operations]
                for shift in range(0, operations + 1):
                    shifted_num = num << shift
                    new_or = current_or | shifted_num
                    dp[operations - shift] = max(dp[operations - shift], new_or)
        
        return max(dp[0:k+1])