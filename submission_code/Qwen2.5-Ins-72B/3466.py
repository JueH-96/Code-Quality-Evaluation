class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            and_val = nums[i]
            for j in range(i, n):
                and_val &= nums[j]
                if and_val == k:
                    count += 1
                elif and_val < k:
                    break
        return count