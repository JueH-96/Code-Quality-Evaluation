class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = []
        suffix_set = set(nums)
        suffix_counts = [0] * n
        current_suffix = set()
        for i in range(n-1, -1, -1):
            current_suffix.add(nums[i])
            suffix_counts[i] = len(current_suffix)
        
        prefix_set = set()
        for i in range(n):
            prefix_set.add(nums[i])
            prefix_distinct = len(prefix_set)
            suffix_distinct = suffix_counts[i+1] if i+1 < n else 0
            result.append(prefix_distinct - suffix_distinct)
        
        return result