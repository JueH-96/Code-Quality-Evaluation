class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        max_beauty = 0
        for i in range(len(nums)):
            target = nums[i]
            current_beauty = 0
            for j in range(len(nums)):
                if abs(nums[j] - target) <= k:
                    current_beauty += 1
            max_beauty = max(max_beauty, current_beauty)
        return max_beauty