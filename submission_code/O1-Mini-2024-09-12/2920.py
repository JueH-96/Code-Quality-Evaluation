from collections import defaultdict
class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        min_steps = float('inf')
        for num, pos in positions.items():
            pos_sorted = sorted(pos)
            max_gap = 0
            k = len(pos_sorted)
            for i in range(k):
                next_i = (i + 1) % k
                if next_i == 0:
                    gap = n - pos_sorted[i] + pos_sorted[0]
                else:
                    gap = pos_sorted[next_i] - pos_sorted[i]
                if gap > max_gap:
                    max_gap = gap
            if max_gap <=1:
                steps =0
            else:
                steps = max_gap //2
            min_steps = min(min_steps, steps)
        
        if len(positions) ==1:
            return 0
        return min_steps