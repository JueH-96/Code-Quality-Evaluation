class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        suffix_unique_counts = [0] * n
        current_suffix_unique = set()
        
        # Populate suffix_unique_counts
        for i in range(n-1, -1, -1):
            suffix_unique_counts[i] = len(current_suffix_unique)
            current_suffix_unique.add(nums[i])
        
        prefix_unique = set()
        diff = []
        
        # Compute the distinct difference array
        for i in range(n):
            prefix_unique.add(nums[i])
            diff.append(len(prefix_unique) - suffix_unique_counts[i])
        
        return diff