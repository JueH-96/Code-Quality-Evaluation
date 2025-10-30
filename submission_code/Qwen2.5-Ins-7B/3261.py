class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [float('inf')] * (1 << 30)
        dp[0] = 0
        
        for num in nums:
            new_dp = dp.copy()
            for mask in range(1 << 30):
                if dp[mask] != float('inf'):
                    new_dp[mask | num] = min(new_dp[mask | num], dp[mask] + 1)
                    for bit in range(30):
                        if mask & (1 << bit) == 0 and (mask | (num & (1 << bit))) == mask:
                            new_dp[mask] = min(new_dp[mask], dp[mask] + 1)
            dp = new_dp
        
        return min(mask for mask in range(1 << 30) if dp[mask] <= k)