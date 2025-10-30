class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr: List[int]) -> bool:
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i + 1]:
                    return False
            return True
        
        n = len(nums)
        count = 0
        
        # Check all subarrays [start, end]
        for start in range(n):
            for end in range(start, n):
                # Remove subarray nums[start:end+1] and check if the result is strictly increasing
                new_arr = nums[:start] + nums[end+1:]
                if is_strictly_increasing(new_arr):
                    count += 1
                    
        return count