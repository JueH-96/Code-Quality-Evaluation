class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 4:
            return 0

        # Find the missing elements
        missing = [i for i in nums if i == -1]
        nums = [i for i in nums if i != -1]
        nums.sort()

        # Replace the missing elements with the minimum and maximum possible values
        min_val, max_val = 1, 10 ** 9
        for x, y in [(min_val, max_val), (max_val, min_val)]:
            for i in range(len(missing)):
                nums[i] = x
                nums[-1 - i] = y

        # Calculate the minimum absolute difference between adjacent elements
        min_diff = float('inf')
        for i in range(1, len(nums)):
            min_diff = min(min_diff, abs(nums[i] - nums[i-1]))

        return min_diff