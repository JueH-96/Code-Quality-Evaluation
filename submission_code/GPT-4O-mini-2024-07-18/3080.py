class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        current_and = nums[0]

        for i in range(n):
            current_and &= nums[i]
            if current_and == 0:
                count += 1
                if i + 1 < n:
                    current_and = nums[i + 1]

        if current_and != 0:
            count += 1

        return count