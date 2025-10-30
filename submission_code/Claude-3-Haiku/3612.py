class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        for i in range(n - 2 * k + 1):
            if all(nums[i] < nums[i+j] for j in range(1, k)):
                if all(nums[i+k] < nums[i+k+j] for j in range(1, k)):
                    return True
        return False