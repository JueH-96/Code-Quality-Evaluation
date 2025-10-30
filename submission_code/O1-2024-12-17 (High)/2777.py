class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        
        for i in range(n):
            # prefix includes nums[0...i]
            prefix_set = set(nums[:i+1])
            # suffix includes nums[i+1...n-1]
            suffix_set = set(nums[i+1:])
            
            diff_val = len(prefix_set) - len(suffix_set)
            diff.append(diff_val)
        
        return diff