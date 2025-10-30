class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        total_sum = sum(nums)
        counts = Counter(nums)
        max_outlier = None
        for x in nums:
            counts[x] -= 1
            s_candidate = total_sum - x
            if s_candidate % 2 == 0:
                s_candidate = s_candidate // 2
                if counts.get(s_candidate, 0) >= 1:
                    if max_outlier is None or x > max_outlier:
                        max_outlier = x
            counts[x] += 1
        return max_outlier