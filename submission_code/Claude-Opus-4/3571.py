class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        from sortedcontainers import SortedList
        
        target = coordinates[k]
        n = len(coordinates)
        
        # Find longest path ending at target
        # We need points with x < target[0] and y < target[1]
        left_points = []
        for i, (x, y) in enumerate(coordinates):
            if x < target[0] and y < target[1]:
                left_points.append((x, y, i))
        
        # Sort by x coordinate
        left_points.sort()
        
        # dp[i] = longest path ending at point i
        dp_left = {}
        
        # Use SortedList to maintain y-coordinates and their max path lengths
        y_to_max_len = SortedList()
        
        for x, y, idx in left_points:
            # Find all points with y' < y
            pos = y_to_max_len.bisect_left((y, 0))
            
            # Get maximum length from all previous points
            max_len = 0
            for i in range(pos):
                max_len = max(max_len, y_to_max_len[i][1])
            
            dp_left[idx] = max_len + 1
            
            # Update or insert current y with its path length
            # Remove any existing entry for this y first
            while True:
                pos = y_to_max_len.bisect_left((y, 0))
                if pos < len(y_to_max_len) and y_to_max_len[pos][0] == y:
                    y_to_max_len.pop(pos)
                else:
                    break
            
            y_to_max_len.add((y, dp_left[idx]))
        
        # Maximum path ending at target
        max_ending_at_k = 0
        for x, y, idx in left_points:
            if x < target[0] and y < target[1]:
                max_ending_at_k = max(max_ending_at_k, dp_left.get(idx, 0))
        
        # Find longest path starting from target
        # We need points with x > target[0] and y > target[1]
        right_points = []
        for i, (x, y) in enumerate(coordinates):
            if x > target[0] and y > target[1]:
                right_points.append((x, y, i))
        
        # Sort by x coordinate in reverse order
        right_points.sort(reverse=True)
        
        dp_right = {}
        y_to_max_len = SortedList()
        
        for x, y, idx in right_points:
            # Find all points with y' > y
            pos = y_to_max_len.bisect_right((y, float('inf')))
            
            # Get maximum length from all previous points
            max_len = 0
            for i in range(pos, len(y_to_max_len)):
                max_len = max(max_len, y_to_max_len[i][1])
            
            dp_right[idx] = max_len + 1
            
            # Update or insert current y with its path length
            while True:
                pos = y_to_max_len.bisect_left((y, 0))
                if pos < len(y_to_max_len) and y_to_max_len[pos][0] == y:
                    y_to_max_len.pop(pos)
                else:
                    break
            
            y_to_max_len.add((y, dp_right[idx]))
        
        # Maximum path starting from target
        max_starting_from_k = 0
        for x, y, idx in right_points:
            if x > target[0] and y > target[1]:
                max_starting_from_k = max(max_starting_from_k, dp_right.get(idx, 0))
        
        # Total length is left + 1 (for k itself) + right
        return max_ending_at_k + 1 + max_starting_from_k