class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        prefix = [0] * len(nums)
        suffix = [0] * len(nums)
        prefix_set = set()
        suffix_set = set()
        res = []
        
        for i in range(len(nums)):
            prefix[i] = len(prefix_set)
            suffix[i] = len(suffix_set)
            prefix_set.add(nums[i])
            suffix_set.add(nums[~i])
        for i in range(len(nums)):
            res.append(prefix[i] - suffix[i])
        return res