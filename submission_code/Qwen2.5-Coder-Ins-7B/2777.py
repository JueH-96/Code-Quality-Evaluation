class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_count = [0] * n
        suffix_count = [0] * n
        prefix_set = set()
        suffix_set = set(nums)
        
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_count[i] = len(prefix_set)
            suffix_set.remove(nums[i])
            suffix_count[i] = len(suffix_set)
        
        diff = [prefix_count[i] - suffix_count[i] for i in range(n)]
        return diff