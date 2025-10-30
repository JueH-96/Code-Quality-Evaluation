class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i+1] for i in range(len(arr)-1))
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                left = nums[:i]
                right = nums[j+1:]
                if is_strictly_increasing(left + right):
                    count += 1
        
        return count