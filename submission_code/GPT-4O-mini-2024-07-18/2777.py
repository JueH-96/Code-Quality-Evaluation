class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n
        
        # Calculate distinct elements in prefix
        prefix_distinct = set()
        for i in range(n):
            prefix_distinct.add(nums[i])
            prefix_count = len(prefix_distinct)
            
            # Calculate distinct elements in suffix
            suffix_distinct = set(nums[i + 1:])
            suffix_count = len(suffix_distinct)
            
            # Calculate the difference
            diff[i] = prefix_count - suffix_count
        
        return diff