class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff_array = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            distinct_prefix_count = len(set(prefix))
            distinct_suffix_count = len(set(suffix))
            diff_array.append(distinct_prefix_count - distinct_suffix_count)
        return diff_array