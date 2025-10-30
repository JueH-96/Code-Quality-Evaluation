class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        mod = 10**9 + 7
        res = 0
        prefix_sum = 0
        
        for num in nums:
            res = (res + num**2 * (prefix_sum + num)) % mod
            prefix_sum = (prefix_sum * 2 + num) % mod
        
        return res