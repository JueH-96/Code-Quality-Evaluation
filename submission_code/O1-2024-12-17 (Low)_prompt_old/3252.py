class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i+1]:
                    return False
            return True
        
        n = len(nums)
        count = 0
        # Enumerate all subarrays [i, j] to remove
        for i in range(n):
            for j in range(i, n):
                # Remove the subarray nums[i:j+1]
                new_array = nums[:i] + nums[j+1:]
                if is_strictly_increasing(new_array):
                    count += 1
        
        return count