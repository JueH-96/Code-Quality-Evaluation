class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        mod = 10 ** 9 + 7
        
        def power(i, j):
            return abs(nums[i] - nums[j])
        
        total = 0
        for i in range(n - k + 1):
            min_power = float('inf')
            for j in range(i, i + k):
                min_power = min(min_power, power(i, j))
            total += min_power
        
        return total % mod