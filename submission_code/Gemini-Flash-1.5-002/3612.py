class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        def is_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i+1]:
                    return False
            return True

        n = len(nums)
        for a in range(n - 2 * k + 1):
            subarray1 = nums[a:a+k]
            subarray2 = nums[a+k:a+2*k]
            if is_increasing(subarray1) and is_increasing(subarray2):
                return True
        return False