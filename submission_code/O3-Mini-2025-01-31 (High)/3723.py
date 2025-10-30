from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total_good = 0
        n = len(nums)
        for i in range(n):
            is_good = True
            # Check the element at i - k, if exists.
            if i - k >= 0 and nums[i] <= nums[i - k]:
                is_good = False
            # Check the element at i + k, if exists.
            if i + k < n and nums[i] <= nums[i + k]:
                is_good = False
            # If the element passed all applicable conditions, add it to the total.
            if is_good:
                total_good += nums[i]
        return total_good

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.sumOfGoodNumbers([1,3,2,1,5,4], 2))  # Output: 12
    # Example 2
    print(sol.sumOfGoodNumbers([2,1], 1))  # Output: 2