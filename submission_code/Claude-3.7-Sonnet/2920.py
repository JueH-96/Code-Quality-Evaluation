from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        min_time = float('inf')
        
        # Get positions for each unique value
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        for num, pos in positions.items():
            pos.sort()
            max_gap = 0
            
            # Calculate gap between consecutive positions
            for i in range(1, len(pos)):
                gap = pos[i] - pos[i-1] - 1
                max_gap = max(max_gap, gap)
            
            # Calculate wraparound gap (from last occurrence to first)
            gap = n - pos[-1] + pos[0] - 1
            max_gap = max(max_gap, gap)
            
            # Time to propagate = ceiling(gap/2)
            time = (max_gap + 1) // 2
            min_time = min(min_time, time)
        
        return min_time