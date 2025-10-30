class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        # Iterate over every subarray of length 3
        for i in range(len(nums) - 2):
            a, b, c = nums[i], nums[i+1], nums[i+2]
            # Instead of checking a + c == b / 2, we check 2*(a+c)==b for accurate integer arithmetic.
            if 2 * (a + c) == b:
                count += 1
        return count