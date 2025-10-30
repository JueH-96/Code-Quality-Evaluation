class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        for a in range(len(nums) - 2 * k + 1):
            if all(nums[j] < nums[j + 1] for j in range(a, a + k - 1)) and \
               all(nums[j] < nums[j + 1] for j in range(a + k, a + 2 * k - 1)):
                return True
        return False