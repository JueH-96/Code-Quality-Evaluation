from collections import Counter

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        cnt = Counter(nums)
        dom = None
        for x, freq in cnt.items():
            if freq * 2 > n:
                dom = x
                break
        total_freq = cnt[dom]
        left_freq = 0
        for i in range(n - 1):
            if nums[i] == dom:
                left_freq += 1
            right_freq = total_freq - left_freq
            if left_freq * 2 > (i + 1) and right_freq * 2 > (n - i - 1):
                return i
        return -1