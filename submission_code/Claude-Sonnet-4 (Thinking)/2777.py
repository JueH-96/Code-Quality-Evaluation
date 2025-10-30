class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        
        for i in range(n):
            # Count distinct elements in prefix nums[0...i]
            prefix_count = len(set(nums[:i+1]))
            
            # Count distinct elements in suffix nums[i+1...n-1]
            suffix_count = len(set(nums[i+1:]))
            
            # Calculate the difference
            diff.append(prefix_count - suffix_count)
        
        return diff