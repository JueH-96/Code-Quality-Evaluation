class Solution:
    def minElement(self, nums: List[int]) -> int:
        def digit_sum(x):
            return sum(int(d) for d in str(x))
        
        return min(digit_sum(num) for num in nums)