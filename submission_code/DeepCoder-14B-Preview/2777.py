class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        diff = []
        for i in range(len(nums)):
            prefix = nums[:i+1]
            prefix_distinct = len(set(prefix))
            suffix = nums[i+1:]
            suffix_distinct = len(set(suffix))
            diff.append(prefix_distinct - suffix_distinct)
        return diff