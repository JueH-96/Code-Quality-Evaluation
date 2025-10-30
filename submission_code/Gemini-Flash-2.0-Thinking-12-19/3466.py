class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            current_and = -1
            for j in range(i, n):
                if current_and == -1:
                    current_and = nums[j]
                else:
                    current_and &= nums[j]
                if current_and == k:
                    count += 1
        return count