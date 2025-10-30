class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        dominant = max(nums, key=nums.count)
        if nums.count(dominant) * 2 > n:
            count = 0
            for i in range(n):
                if nums[i] == dominant:
                    count += 1
                if count * 2 > i + 1 and (n - i - 1) * 2 > n - i - 1:
                    return i
        return -1