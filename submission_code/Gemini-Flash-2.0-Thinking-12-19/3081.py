from collections import Counter

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        counts = Counter(nums)
        distinct_nums = sorted(counts.keys())
        num_counts = [counts[x] for x in distinct_nums]
        l, r = 0, len(distinct_nums) - 1
        while l < r:
            pairs_formed = min(num_counts[l], num_counts[r])
            num_counts[l] -= pairs_formed
            num_counts[r] -= pairs_formed
            if num_counts[l] == 0:
                l += 1
            if num_counts[r] == 0:
                r -= 1
        final_length = sum(num_counts)
        return final_length