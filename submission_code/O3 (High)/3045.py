class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Already sorted?
        if n <= 1 or nums == sorted(nums):
            return 0
        
        # Position of the smallest element
        min_idx = nums.index(min(nums))
        
        # Check that the circular sequence is strictly increasing
        for i in range(1, n):
            prev = nums[(min_idx + i - 1) % n]
            curr = nums[(min_idx + i) % n]
            if curr < prev:          # not increasing ⇒ impossible
                return -1
        
        # Minimal right shifts needed (0 … n-1)
        return (n - min_idx) % n