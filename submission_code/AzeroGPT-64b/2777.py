class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        prefix_set = set()
        suffix_set = set(nums)
        result = []
        
        for i, num in enumerate(nums):
            prefix_set.add(num)
            suffix_set.remove(num)
            result.append(len(prefix_set) - len(suffix_set))
            
        return result