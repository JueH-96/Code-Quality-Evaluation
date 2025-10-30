class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums)):
            current_and = nums[i]
            if current_and == k:
                count += 1
            for j in range(i + 1, len(nums)):
                current_and &= nums[j]
                if current_and == k:
                    count += 1
                elif current_and < k:
                    break
        return count