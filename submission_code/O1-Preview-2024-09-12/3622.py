class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        from bisect import bisect_left, bisect_right
        from collections import Counter
        
        counts = Counter(nums)
        sorted_nums = sorted(counts.keys())
        total_counts = []
        prefix_sums = []
        total = 0

        for num in sorted_nums:
            cnt = counts[num]
            total_counts.append(cnt)
            total += cnt
            prefix_sums.append(total)
        
        max_freq = 0
        
        for i, x in enumerate(sorted_nums):
            left = bisect_left(sorted_nums, x - k)
            right = bisect_right(sorted_nums, x + k) - 1  # right index inclusive
            total_counts_in_range = prefix_sums[right]
            if left > 0:
                total_counts_in_range -= prefix_sums[left - 1]
            adjustments_needed = total_counts_in_range - counts[x]
            if adjustments_needed <= numOperations:
                max_freq = max(max_freq, total_counts_in_range)
        
        return max_freq