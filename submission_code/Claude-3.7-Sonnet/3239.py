class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If x and y are already equal, no operations needed
        if x == y:
            return 0
        
        # Use BFS to find the shortest path
        from collections import deque
        
        # Use a set to keep track of visited numbers
        visited = set([x])
        # Queue stores (number, operations_so_far)
        queue = deque([(x, 0)])
        
        while queue:
            num, ops = queue.popleft()
            
            # Try all four operations
            next_states = [
                num - 1,  # Decrement by 1
                num + 1   # Increment by 1
            ]
            
            # Add division operations if applicable
            if num % 11 == 0:
                next_states.append(num // 11)
            if num % 5 == 0:
                next_states.append(num // 5)
            
            # Process each possible next state
            for next_num in next_states:
                # If we've reached the target
                if next_num == y:
                    return ops + 1
                
                # If this is a valid state we haven't seen before
                if next_num > 0 and next_num not in visited:
                    visited.add(next_num)
                    queue.append((next_num, ops + 1))
        
        # This line shouldn't be reached as there's always a solution
        return -1