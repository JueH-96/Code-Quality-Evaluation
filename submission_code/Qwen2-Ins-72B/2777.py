class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_set = set()
        suffix_set = set(nums)
        result = []
        
        for i in range(n):
            prefix_set.add(nums[i])
            suffix_set.remove(nums[i])
            result.append(len(prefix_set) - len(suffix_set))
            
        return result