class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n):
            total = 0
            while n > 0:
                total += n % 10
                n //= 10
            return total
        
        # Replace each element with sum of its digits
        for i in range(len(nums)):
            nums[i] = sum_of_digits(nums[i])
        
        # Return minimum element
        return min(nums)