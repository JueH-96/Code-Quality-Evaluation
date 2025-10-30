class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        positions = defaultdict(list)
        
        # Group positions by value
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        min_time = float('inf')
        
        for pos_list in positions.values():
            if len(pos_list) == n:  # All positions have this value
                return 0
            
            pos_list.sort()
            k = len(pos_list)
            max_distance = 0
            
            # Check each arc between consecutive positions
            for i in range(k):
                start = pos_list[i]
                end = pos_list[(i + 1) % k]  # Next position (circular)
                
                # Calculate arc length (positions to fill)
                if start < end:
                    arc_length = end - start - 1
                else:  # Wrapping around
                    arc_length = n - start - 1 + end
                
                # Time needed for this arc
                arc_max_distance = (arc_length + 1) // 2
                max_distance = max(max_distance, arc_max_distance)
            
            min_time = min(min_time, max_distance)
        
        return min_time