class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0

        # Helper function to check if a subarray is strictly increasing
        def is_strictly_increasing(arr):
            for k in range(1, len(arr)):
                if arr[k-1] >= arr[k]:
                    return False
            return True

        # Iterate over all possible subarrays to remove (i to j)
        for i in range(n):
            for j in range(i, n):
                # Define left and right parts
                left = nums[0:i]
                right = nums[j+1:n]
                
                # Check if left is strictly increasing
                left_ok = is_strictly_increasing(left)
                # Check if right is strictly increasing
                right_ok = is_strictly_increasing(right)
                
                if left_ok and right_ok:
                    if i == 0 and j == n-1:
                        # Removing the entire array, which is allowed
                        count += 1
                    elif i == 0:
                        # Left is empty, only need right to be strictly increasing
                        count += 1
                    elif j == n-1:
                        # Right is empty, only need left to be strictly increasing
                        count += 1
                    else:
                        # Both left and right are non-empty, check last of left < first of right
                        if left[-1] < right[0]:
                            count += 1
        return count