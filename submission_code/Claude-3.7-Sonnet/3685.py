class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums) - 2):  # Iterate through all possible starting positions
            a, b, c = nums[i], nums[i + 1], nums[i + 2]
            if 2 * (a + c) == b:  # Check if sum of first and third equals half of middle
                count += 1
        return count