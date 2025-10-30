class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))
        
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                remaining = nums[:i] + nums[j+1:]
                if is_strictly_increasing(remaining):
                    count += 1
        
        return count