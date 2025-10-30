class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        nums.sort()
        mod = 10 ** 9 + 7
        ans = 0
        total = 0
        for num in nums:
            ans = (ans + num * num * num % mod) % mod
            ans = (ans + num * num * total % mod) % mod
            total = (total * 2 + num) % mod
        return ans