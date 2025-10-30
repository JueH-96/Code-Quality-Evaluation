class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if 2 * k > n:
            return False

        def is_strictly_increasing(subarray):
            for i in range(len(subarray) - 1):
                if subarray[i] >= subarray[i + 1]:
                    return False
            return True

        for a in range(n - 2 * k + 1):
            subarray1 = nums[a:a + k]
            subarray2 = nums[a + k:a + 2 * k]
            if is_strictly_increasing(subarray1) and is_strictly_increasing(subarray2):
                return True
        return False