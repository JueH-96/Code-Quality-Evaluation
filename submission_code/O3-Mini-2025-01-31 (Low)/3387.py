from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        mid = n // 2
        ops = 0
        
        # For positions to the right of the median (including median),
        # ensuring those values are not below k.
        for i in range(mid, n):
            if nums[i] < k:
                ops += k - nums[i]
        
        # For positions to the left of the median (including median),
        # ensuring those values are not above k.
        for i in range(0, mid + 1):
            if nums[i] > k:
                ops += nums[i] - k
        
        return ops

# Example test cases
if __name__ == '__main__':
    sol = Solution()
    print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 4))  # Output: 2
    print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 7))  # Output: 3
    print(sol.minOperationsToMakeMedianK([1,2,3,4,5,6], 4))  # Output: 0