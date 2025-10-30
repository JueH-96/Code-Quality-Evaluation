class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        diff = []
        for i in range(len(nums)):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            prefix_count = len(set(prefix))
            suffix_count = len(set(suffix))
            diff.append(prefix_count - suffix_count)
        return diff