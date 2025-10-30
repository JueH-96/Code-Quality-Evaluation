from math import comb

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = 1
        for num in nums:
            count *= comb(num + 1, 2)
            count %= MOD
        return count