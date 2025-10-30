class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sumDigits(n):
            return sum(int(d) for d in str(n))
        
        return min(sumDigits(n) for n in nums)