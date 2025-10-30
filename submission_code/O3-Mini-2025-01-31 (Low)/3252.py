from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        def is_strictly_increasing(arr: List[int]) -> bool:
            for k in range(len(arr) - 1):
                if arr[k] >= arr[k+1]:
                    return False
            return True
        
        # iterate over all possible contiguous subarrays to remove (non-empty)
        for i in range(n):
            for j in range(i, n):
                # removed subarray is from index i to j; remaining array:
                remaining = nums[:i] + nums[j+1:]
                if is_strictly_increasing(remaining):
                    count += 1
        
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    nums1 = [1,2,3,4]
    print(sol.incremovableSubarrayCount(nums1))  # Expected output: 10

    # Example 2
    nums2 = [6,5,7,8]
    print(sol.incremovableSubarrayCount(nums2))  # Expected output: 7

    # Example 3
    nums3 = [8,7,6,6]
    print(sol.incremovableSubarrayCount(nums3))  # Expected output: 3