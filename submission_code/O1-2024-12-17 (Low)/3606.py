class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(num):
            return sum(int(d) for d in str(num))
        
        return min(sum_of_digits(n) for n in nums)