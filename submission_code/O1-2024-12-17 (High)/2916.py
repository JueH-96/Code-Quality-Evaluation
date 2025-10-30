class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # If array length is 1 or 2, we can always split into individual elements
        if n <= 2:
            return True
        
        # For length >= 3, check if any adjacent pair sums to at least m
        # If so, the answer is True; otherwise, False
        for i in range(n - 1):
            if nums[i] + nums[i + 1] >= m:
                return True
        return False