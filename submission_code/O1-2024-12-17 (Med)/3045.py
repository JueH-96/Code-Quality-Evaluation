class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        
        # Try 0 to n-1 right shifts
        for k in range(n):
            # Generate the array after k right shifts
            rotated = [nums[(i - k) % n] for i in range(n)]
            if rotated == sorted_nums:
                return k
        
        return -1