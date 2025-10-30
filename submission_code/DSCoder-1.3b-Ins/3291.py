class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Count the number of ones in the binary representation of all numbers
        # If the count is not equal to the length of the array, it is impossible to sort the array
        # If the count is equal to the length of the array, the array is already sorted
        count_ones = sum(format(n, 'b').count('1') for n in nums)
        return count_ones == len(nums)