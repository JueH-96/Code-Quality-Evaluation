class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x == y:
            return 0
        
        # Use BFS to find shortest path
        queue = deque([(x, 0)])  # (current_value, operations_count)
        visited = set([x])
        
        # Set upper bound for search - need to allow going beyond max(x,y)
        # to find favorable numbers for division
        upper_bound = 20000  # Safe bound given constraints
        
        while queue:
            curr, ops = queue.popleft()
            
            # Try all 4 possible operations
            next_states = []
            
            # Operation 1: Divide by 11 if divisible
            if curr % 11 == 0:
                next_states.append(curr // 11)
            
            # Operation 2: Divide by 5 if divisible
            if curr % 5 == 0:
                next_states.append(curr // 5)
            
            # Operation 3: Decrement by 1
            if curr > 0:
                next_states.append(curr - 1)
            
            # Operation 4: Increment by 1
            next_states.append(curr + 1)
            
            # Process all possible next states
            for next_val in next_states:
                if next_val == y:
                    return ops + 1
                
                # Only explore unvisited states within bounds
                if next_val not in visited and 0 <= next_val <= upper_bound:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))
        
        return -1  # Should never reach here given valid inputs