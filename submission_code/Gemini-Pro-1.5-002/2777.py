class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            
            prefix_distinct = len(set(prefix))
            suffix_distinct = len(set(suffix))
            
            diff.append(prefix_distinct - suffix_distinct)
        return diff