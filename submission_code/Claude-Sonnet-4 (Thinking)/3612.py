class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        
        def is_strictly_increasing(start, length):
            for i in range(start, start + length - 1):
                if nums[i] >= nums[i + 1]:
                    return False
            return True
        
        for a in range(n - 2 * k + 1):
            if is_strictly_increasing(a, k) and is_strictly_increasing(a + k, k):
                return True
        
        return False