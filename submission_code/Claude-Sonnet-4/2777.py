class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Pre-compute suffix distinct counts
        suffix_distinct = [0] * n
        suffix_set = set()
        
        # Fill suffix_distinct array from right to left
        for i in range(n-2, -1, -1):
            suffix_set.add(nums[i+1])
            suffix_distinct[i] = len(suffix_set)
        
        # Compute result array
        result = []
        prefix_set = set()
        
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_distinct = len(prefix_set)
            diff = prefix_distinct - suffix_distinct[i]
            result.append(diff)
        
        return result