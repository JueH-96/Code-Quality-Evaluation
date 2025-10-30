class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        ans = 0
        for i in range(len(nums)):
            diff = target[i] - nums[i]
            if diff > 0:
                ans += diff
                nums[i] = target[i]
                for j in range(i+1, len(nums)):
                    if nums[j] < target[j] and nums[j] + diff >= target[j]:
                        ans -= target[j] - nums[j]
                        nums[j] = target[j]
                    else:
                        nums[j] += diff
            elif diff < 0:
                ans -= diff
                nums[i] = target[i]
                for j in range(i+1, len(nums)):
                    if nums[j] > target[j] and nums[j] + diff <= target[j]:
                        ans += nums[j] - target[j]
                        nums[j] = target[j]
                    else:
                        nums[j] += diff
        return ans