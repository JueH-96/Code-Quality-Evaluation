class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        left = 0
        cnt = defaultdict(int)
        maxf = 0

        for right, num in enumerate(nums):
            cnt[num] += 1
            maxf = max(maxf, cnt[num])
            if right - left + 1 - maxf > k:
                cnt[nums[left]] -= 1
                left += 1
        return maxf