from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: list) -> int:
        total = sum(nums)
        freq = Counter(nums)
        max_outlier = -10**18
        for num in nums:
            diff = total - num
            if diff % 2 != 0:
                continue
            s = diff // 2
            if s in freq:
                if num == s:
                    if freq[s] >= 2:
                        if num > max_outlier:
                            max_outlier = num
                else:
                    if num > max_outlier:
                        max_outlier = num
        return max_outlier