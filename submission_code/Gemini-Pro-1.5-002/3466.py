class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            current_and = nums[i]
            if current_and == k:
                count += 1
            for j in range(i + 1, n):
                current_and &= nums[j]
                if current_and == k:
                    count += 1
        return count