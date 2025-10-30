class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n: int) -> int:
            return sum(int(d) for d in str(n))
        
        # Replace each element with the sum of its digits
        replaced = [sum_of_digits(num) for num in nums]
        # Return the minimum element of the replaced list
        return min(replaced)