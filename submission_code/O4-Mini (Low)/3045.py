from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # If already sorted, no shifts needed
        if nums == sorted(nums):
            return 0
        
        target = sorted(nums)
        # Try to find a rotation point i such that
        # rotating nums left by i positions equals target.
        for i in range(n):
            # Left rotation by i: nums[i:] + nums[:i]
            if nums[i:] + nums[:i] == target:
                # A left rotation by i is a right rotation by (n - i)
                return (n - i) % n
        
        # No valid rotation found
        return -1

# Example usage:
# sol = Solution()
# print(sol.minimumRightShifts([3,4,5,1,2]))  # Output: 2
# print(sol.minimumRightShifts([1,3,5]))      # Output: 0
# print(sol.minimumRightShifts([2,1,4]))      # Output: -1