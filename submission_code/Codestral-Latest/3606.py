class Solution:
    def minElement(self, nums: List[int]) -> int:
        def sum_of_digits(n):
            return sum(int(digit) for digit in str(n))

        # Replace each element with the sum of its digits
        nums = [sum_of_digits(num) for num in nums]

        # Return the minimum element in the modified nums
        return min(nums)