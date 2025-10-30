class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        suffix_distinct_count = len(set(nums))
        prefix_distinct = set()
        diff = []
        
        for i in range(n):
            prefix_distinct.add(nums[i])
            if i < n - 1 and nums[i + 1] in prefix_distinct:
                suffix_distinct_count -= 1
            diff.append(len(prefix_distinct) - suffix_distinct_count)
        
        return diff