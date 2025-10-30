class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x == y:
            return 0
        
        # BFS approach
        queue = deque([(x, 0)])  # (current_value, steps)
        visited = {x}
        
        while queue:
            curr, steps = queue.popleft()
            
            # Try all 4 operations
            next_values = []
            
            # Operation 1: Divide by 11 if multiple of 11
            if curr % 11 == 0:
                next_values.append(curr // 11)
            
            # Operation 2: Divide by 5 if multiple of 5
            if curr % 5 == 0:
                next_values.append(curr // 5)
            
            # Operation 3: Decrement by 1
            next_values.append(curr - 1)
            
            # Operation 4: Increment by 1
            next_values.append(curr + 1)
            
            for next_val in next_values:
                if next_val == y:
                    return steps + 1
                
                # Add bounds to prevent infinite exploration
                # We shouldn't need to go below 0 or above 2*max(x,y)
                if next_val not in visited and 0 <= next_val <= 2 * max(x, y):
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
        
        return -1  # Should never reach here given the constraints