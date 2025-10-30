class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [0] * n
        suffix = [0] * (n + 1)
        
        # Calculate distinct elements in suffix
        distinct_set = set()
        for i in range(n - 1, -1, -1):
            distinct_set.add(nums[i])
            suffix[i] = len(distinct_set)
        
        # Calculate distinct elements in prefix and compute difference
        distinct_set.clear()
        result = []
        for i in range(n):
            distinct_set.add(nums[i])
            prefix_count = len(distinct_set)
            suffix_count = suffix[i + 1]
            result.append(prefix_count - suffix_count)
        
        return result