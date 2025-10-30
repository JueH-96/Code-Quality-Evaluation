class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_increasing(arr):
            if len(arr) <= 1:
                return True
            for k in range(len(arr) - 1):
                if arr[k] >= arr[k + 1]:
                    return False
            return True
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                new_arr = nums[:i] + nums[j+1:]
                if is_increasing(new_arr):
                    count += 1
        return count