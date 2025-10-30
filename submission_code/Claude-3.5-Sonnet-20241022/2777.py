class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        
        for i in range(n):
            # Count distinct elements in prefix [0...i]
            prefix = len(set(nums[:i+1]))
            
            # Count distinct elements in suffix [i+1...n-1]
            suffix = len(set(nums[i+1:])) if i < n-1 else 0
            
            diff.append(prefix - suffix)
            
        return diff