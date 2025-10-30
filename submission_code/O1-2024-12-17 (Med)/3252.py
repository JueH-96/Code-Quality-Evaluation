class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr: List[int]) -> bool:
            for i in range(1, len(arr)):
                if arr[i] <= arr[i - 1]:
                    return False
            return True
        
        n = len(nums)
        count = 0
        
        # Iterate over all possible subarrays [i, j]
        for i in range(n):
            for j in range(i, n):
                # Left part (before subarray to remove) and right part (after subarray)
                left = nums[:i]
                right = nums[j + 1:]
                
                # Check if left and right parts are each strictly increasing
                if is_strictly_increasing(left) and is_strictly_increasing(right):
                    # If both parts are non-empty, also check the bridge condition
                    if not left or not right or left[-1] < right[0]:
                        count += 1
        
        return count