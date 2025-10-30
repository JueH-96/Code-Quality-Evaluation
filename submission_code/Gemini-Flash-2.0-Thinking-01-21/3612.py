class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if 2 * k > n:
            return False

        def is_strictly_increasing(arr):
            for i in range(1, len(arr)):
                if arr[i] <= arr[i - 1]:
                    return False
            return True

        for a in range(n - 2 * k + 1):
            subarray1 = nums[a:a + k]
            subarray2 = nums[a + k:a + 2 * k]
            if is_strictly_increasing(subarray1) and is_strictly_increasing(subarray2):
                return True
        return False