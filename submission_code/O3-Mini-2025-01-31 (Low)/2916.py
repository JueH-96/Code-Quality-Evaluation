from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        # If the array is a single element, we already have it split into one subarray.
        if n == 1:
            return True
        
        # The key observation is that if there's at least one pair of adjacent elements
        # whose sum is at least m, then it is always possible to perform the required sequence of splits.
        # In each split, one of the subarrays can be a single element while the other subarray remains valid (with sum >= m),
        # allowing us to ultimately split the entire array into n pieces.
        for i in range(n - 1):
            if nums[i] + nums[i+1] >= m:
                return True
        
        return False

# You can test the solution with the provided examples:
if __name__ == "__main__":
    sol = Solution()
    print(sol.canSplitArray([2, 2, 1], 4))  # Expected output: True
    print(sol.canSplitArray([2, 1, 3], 5))  # Expected output: False
    print(sol.canSplitArray([2, 3, 3, 2, 3], 6))  # Expected output: True