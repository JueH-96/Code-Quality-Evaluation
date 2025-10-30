class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        # If x is already y, then no operations are required
        if x == y:
            return 0
        
        # We will use BFS to find the minimum operations
        # Define a reasonable bound for BFS (since x,y <= 10^4, we pick a bit higher to safely include increments)
        MAX_VAL = 20000
        
        visited = [False] * (MAX_VAL + 1)
        queue = deque()
        
        # Initialize BFS with the starting point
        queue.append((x, 0))
        visited[x] = True
        
        while queue:
            current, steps = queue.popleft()
            
            # Generate possible next states
            next_states = []
            
            # Operation 1: Divide by 11 if multiple of 11
            if current % 11 == 0:
                next_states.append(current // 11)
            
            # Operation 2: Divide by 5 if multiple of 5
            if current % 5 == 0:
                next_states.append(current // 5)
            
            # Operation 3: Decrement by 1 (if > 1, since we only consider positive integers)
            if current > 1:
                next_states.append(current - 1)
            
            # Operation 4: Increment by 1 (within the bound)
            if current < MAX_VAL:
                next_states.append(current + 1)
            
            # Traverse the next states
            for nxt in next_states:
                if 1 <= nxt <= MAX_VAL and not visited[nxt]:
                    if nxt == y:
                        return steps + 1
                    visited[nxt] = True
                    queue.append((nxt, steps + 1))
        
        # Should not reach here if the BFS is conducted properly, but return -1 as fallback
        return -1