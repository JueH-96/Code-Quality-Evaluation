class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Helper function to check if the array is strictly increasing
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))
        
        # Iterate over all possible subarrays
        for i in range(n):
            for j in range(i, n):
                # Remove the subarray nums[i:j+1] and check if the remaining array is strictly increasing
                if is_strictly_increasing(nums[:i] + nums[j+1:]):
                    count += 1
        
        return count