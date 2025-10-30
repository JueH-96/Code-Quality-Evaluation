class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        nums_copy = list(nums)
        for l, r in queries:
            min_val = float('inf')
            for i in range(l, r + 1):
                min_val = min(min_val, nums_copy[i])
            
            for i in range(l, r + 1):
                nums_copy[i] -= min_val

        for num in nums_copy:
            if num != 0:
                return False
        return True