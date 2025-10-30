class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Helper function to check if an array is strictly increasing
        def is_strictly_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i + 1]:
                    return False
            return True
        
        count = 0
        # Try removing every possible subarray [i..j]
        for i in range(n):
            for j in range(i, n):
                # Remove subarray and form new array
                new_array = nums[:i] + nums[j+1:]
                if is_strictly_increasing(new_array):
                    count += 1
        
        return count