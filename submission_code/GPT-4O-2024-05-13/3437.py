from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = [0] * n
        
        for i in range(n):
            dp[i] = power[i]
            for j in range(i):
                if abs(power[i] - power[j]) > 2:
                    dp[i] = max(dp[i], dp[j] + power[i])
        
        return max(dp)

# Example usage:
# sol = Solution()
# print(sol.maximumTotalDamage([1,1,3,4]))  # Output: 6
# print(sol.maximumTotalDamage([7,1,6,6]))  # Output: 13