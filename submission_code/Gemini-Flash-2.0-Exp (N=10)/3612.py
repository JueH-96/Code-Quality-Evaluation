class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if n < 2 * k:
            return False

        def is_strictly_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i+1]:
                    return False
            return True

        for a in range(n - 2 * k + 1):
            b = a + k
            subarray_a = nums[a:a+k]
            subarray_b = nums[b:b+k]
            if is_strictly_increasing(subarray_a) and is_strictly_increasing(subarray_b):
                return True
        
        return False