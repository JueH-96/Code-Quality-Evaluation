from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums.sort()
        n = len(nums)
        F = [0] * n
        F[0] = nums[0] % mod
        for j in range(1, n):
            F[j] = (nums[j] + 2 * F[j-1] - nums[j-1]) % mod
        
        total = 0
        for j in range(n):
            total = (total + (nums[j] * nums[j] % mod) * F[j]) % mod
        
        return total