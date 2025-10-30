class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        
        # Group indices by value
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        min_seconds = float('inf')
        
        # For each unique value, calculate the time needed to spread everywhere
        for num, indices in positions.items():
            if len(indices) == n:
                # Already all equal
                return 0
            
            # Find maximum gap between consecutive occurrences
            max_gap = 0
            
            # Check gaps between consecutive indices
            for i in range(len(indices)):
                # Gap from current index to next index (circular)
                next_idx = (i + 1) % len(indices)
                gap = (indices[next_idx] - indices[i] + n) % n
                max_gap = max(max_gap, gap)
            
            # Time needed is ceil(max_gap / 2)
            seconds_needed = (max_gap + 1) // 2
            min_seconds = min(min_seconds, seconds_needed)
        
        return min_seconds