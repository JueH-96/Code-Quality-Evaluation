class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_distinct = [0] * n
        suffix_distinct = [0] * n
        prefix_set = set()
        suffix_set = set()

        # Calculate distinct elements in the prefix
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_distinct[i] = len(prefix_set)

        # Calculate distinct elements in the suffix
        for i in range(n-1, -1, -1):
            suffix_set.add(nums[i])
            suffix_distinct[i] = len(suffix_set)

        # Calculate the distinct difference array
        diff = [0] * n
        for i in range(n):
            diff[i] = prefix_distinct[i] - suffix_distinct[i]

        return diff