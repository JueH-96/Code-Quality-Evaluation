class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        # If x is already equal to y, no operations are needed
        if x == y:
            return 0
        
        # We will perform a BFS to find the minimum number of operations
        queue = deque([(x, 0)])
        visited = set([x])
        
        # Define an upper bound so we don't explore too large numbers
        # 2*10^4 is sufficient for searching given the constraints
        UPPER_BOUND = 20000
        
        while queue:
            current, steps = queue.popleft()
            
            # Try the four possible operations:
            # 1) Decrement by 1
            if current - 1 >= 1:
                if current - 1 == y:
                    return steps + 1
                if (current - 1) not in visited:
                    visited.add(current - 1)
                    queue.append((current - 1, steps + 1))
            
            # 2) Increment by 1
            if current + 1 <= UPPER_BOUND:
                if current + 1 == y:
                    return steps + 1
                if (current + 1) not in visited:
                    visited.add(current + 1)
                    queue.append((current + 1, steps + 1))
            
            # 3) Divide by 11 if current is multiple of 11
            if current % 11 == 0:
                nxt = current // 11
                if nxt == y:
                    return steps + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, steps + 1))
            
            # 4) Divide by 5 if current is multiple of 5
            if current % 5 == 0:
                nxt = current // 5
                if nxt == y:
                    return steps + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, steps + 1))
        
        # Theoretically, we should always be able to reach y.
        # Return -1 only if something went wrong (should not happen under valid constraints).
        return -1