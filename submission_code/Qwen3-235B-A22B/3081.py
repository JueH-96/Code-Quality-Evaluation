class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        i = 0
        j = (n + 1) // 2
        count = 0
        
        while i < (n + 1) // 2 and j < n:
            if nums[i] < nums[j]:
                count += 1
                i += 1
                j += 1
            else:
                j += 1
        
        return n - 2 * count