class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        n = len(nums)
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)
        
        min_time = float('inf')
        for num in positions:
            idxs = positions[num]
            idxs.sort()
            max_gap = 0
            for i in range(len(idxs)):
                if i == len(idxs) -1:
                    # Circular gap
                    gap = idxs[0] + n - idxs[i]
                else:
                    gap = idxs[i+1] - idxs[i]
                time_needed = gap //2
                max_gap = max(max_gap, time_needed)
            min_time = min(min_time, max_gap)
        return min_time