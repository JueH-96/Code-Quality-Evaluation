class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff_array = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            prefix_distinct_count = len(set(prefix))
            suffix_distinct_count = len(set(suffix))
            diff = prefix_distinct_count - suffix_distinct_count
            diff_array.append(diff)
        return diff_array