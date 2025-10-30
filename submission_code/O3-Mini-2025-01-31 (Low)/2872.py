from typing import List
class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We can start from the rightmost element, and iterate backwards while merging.
        # If a number is less than or equal to the cumulative sum (from right),
        # we can combine it, otherwise, we "restart" the cumulative sum.
        result = nums[-1]
        # iterate from right second last element to the beginning.
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= result:
                result += nums[i]
            else:
                result = nums[i]
        return result

# Example usage
if __name__ == '__main__':
    sol = Solution()
    print(sol.maxArrayValue([2,3,7,9,3]))  # Expected output: 21
    print(sol.maxArrayValue([5,3,3]))      # Expected output: 11