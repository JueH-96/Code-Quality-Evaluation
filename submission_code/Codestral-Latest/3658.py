class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Extract the non-missing elements
        non_missing = [num for num in nums if num != -1]

        # If all elements are missing, the minimum difference is 0
        if not non_missing:
            return 0

        # Sort the non-missing elements
        non_missing.sort()

        # Calculate the differences between consecutive elements
        differences = [non_missing[i + 1] - non_missing[i] for i in range(len(non_missing) - 1)]

        # If there are no missing elements, return the minimum difference
        if -1 not in nums:
            return min(differences)

        # Calculate the minimum difference by replacing missing elements with the smallest and largest non-missing elements
        min_val, max_val = non_missing[0], non_missing[-1]
        min_diff = float('inf')

        for i in range(len(nums) - 1):
            if nums[i] == -1 or nums[i + 1] == -1:
                min_diff = min(min_diff, abs(max_val - min_val))

        return min_diff