class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to hold positions of each distinct value
        positions_map = defaultdict(list)
        
        n = len(nums)
        
        # Collect the indices for each distinct value
        for i, val in enumerate(nums):
            positions_map[val].append(i)
        
        min_seconds = float('inf')
        
        # For each distinct value, compute the largest gap in circular manner
        for val, positions in positions_map.items():
            positions.sort()
            
            max_gap = 0
            # Check gap between consecutive occurrences
            for i in range(1, len(positions)):
                gap = positions[i] - positions[i-1] - 1
                max_gap = max(max_gap, gap)
            
            # Check wrap-around gap
            wrap_gap = (positions[0] + n) - positions[-1] - 1
            max_gap = max(max_gap, wrap_gap)
            
            # The number of steps to fill the largest gap
            # each gap of length G needs (G+1)//2 seconds
            seconds_needed = (max_gap + 1) // 2
            
            min_seconds = min(min_seconds, seconds_needed)
        
        return min_seconds