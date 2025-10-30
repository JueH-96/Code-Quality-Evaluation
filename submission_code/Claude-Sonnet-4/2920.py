class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        if n == 1:
            return 0
        
        # Group positions by value
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        min_seconds = float('inf')
        
        # Try each unique value as the target
        for value, pos_list in positions.items():
            if len(pos_list) == n:  # All elements are already the same
                return 0
            
            # Calculate maximum gap between consecutive positions
            max_gap = 0
            
            # Check gaps between consecutive positions
            for i in range(len(pos_list)):
                next_i = (i + 1) % len(pos_list)
                if next_i == 0:  # Wrap around case
                    gap = n - pos_list[i] + pos_list[0]
                else:
                    gap = pos_list[next_i] - pos_list[i]
                max_gap = max(max_gap, gap)
            
            # Time needed is half the maximum gap (rounded down)
            # because spreading happens from both ends of the gap
            seconds_needed = max_gap // 2
            min_seconds = min(min_seconds, seconds_needed)
        
        return min_seconds