class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n
        
        for i in range(n):
            prefix_set = set(nums[:i+1])
            suffix_set = set(nums[i+1:])
            diff[i] = len(prefix_set) - len(suffix_set)
        
        return diff