class Solution:
    def minElement(self, nums: List[int]) -> int:
        # Helper function to compute sum of digits of a number
        def sum_of_digits(n: int) -> int:
            total = 0
            while n > 0:
                total += n % 10
                n //= 10
            return total
        
        # Replace each element with the sum of its digits
        replaced_nums = [sum_of_digits(num) for num in nums]
        
        # Return the minimum element
        return min(replaced_nums)