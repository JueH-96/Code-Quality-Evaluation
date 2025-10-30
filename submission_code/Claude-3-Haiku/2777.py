class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = [0] * n
        
        # Calculate the number of distinct elements in the prefix
        prefix_distinct = set()
        for i in range(n):
            prefix_distinct.add(nums[i])
            diff[i] = len(prefix_distinct) - self.count_distinct(nums, i+1, n-1)
        
        return diff
    
    def count_distinct(self, nums: List[int], start: int, end: int) -> int:
        if start > end:
            return 0
        
        distinct = set()
        for i in range(start, end+1):
            distinct.add(nums[i])
        return len(distinct)