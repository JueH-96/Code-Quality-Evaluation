class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(x: int) -> int:
            return sum(int(digit) for digit in str(x))
        
        transformed = [sum_of_digits(n) for n in nums]
        return min(transformed)