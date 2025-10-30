class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        from collections import defaultdict
        num_positions = defaultdict(list)
        for idx, num in enumerate(nums):
            if num in num_positions:
                num_positions[num][1] = idx
            else:
                num_positions[num] = [idx, idx]
        intervals = list(num_positions.values())
        intervals.sort()
        merged_intervals = []
        for interval in intervals:
            if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
        m = len(merged_intervals)
        if m == 0:
            return 0
        return pow(2, m -1 , mod)