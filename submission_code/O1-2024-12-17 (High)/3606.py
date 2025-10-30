class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(x: int) -> int:
            return sum(int(d) for d in str(x))
        
        return min(sum_of_digits(num) for num in nums)