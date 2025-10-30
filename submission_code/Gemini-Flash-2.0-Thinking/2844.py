class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        n = len(nums)
        sum_of_squares = 0
        for i in range(1, n + 1):
            if n % i == 0:
                # Since nums is 1-indexed, the element at index i is nums[i-1]
                special_element = nums[i - 1]
                sum_of_squares += special_element * special_element
        return sum_of_squares