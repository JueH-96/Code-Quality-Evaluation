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
        
        min_time = n  # Initialize with a large value
        
        for num, pos_list in positions.items():
            k = len(pos_list)
            if k == n:
                # All elements are already the same
                return 0
            
            # Calculate maximum gap between consecutive occurrences
            max_gap = 0
            for i in range(k):
                if i == k - 1:
                    # Circular gap from last position to first
                    gap = n - pos_list[i] + pos_list[0]
                else:
                    gap = pos_list[i+1] - pos_list[i]
                max_gap = max(max_gap, gap)
            
            # Time needed is half the maximum gap
            time = max_gap // 2
            min_time = min(min_time, time)
        
        return min_time