class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        from collections import deque
        
        if x == y:
            return 0
        
        # We choose an upper bound that is safely above the maximum values we might need.
        # Given x, y <= 10^4, going to 20000 is sufficient to handle increments/decrements
        # while still allowing divisions when beneficial.
        MAX_VAL = 20000
        
        queue = deque([(x, 0)])  # (current_value, number_of_operations)
        visited = set([x])
        
        while queue:
            val, steps = queue.popleft()
            
            # Generate possible next values based on the allowed operations:
            candidates = []
            # Decrement
            if val - 1 >= 1:
                candidates.append(val - 1)
            # Increment
            if val + 1 <= MAX_VAL:
                candidates.append(val + 1)
            # Divide by 5 if multiple of 5
            if val % 5 == 0:
                candidates.append(val // 5)
            # Divide by 11 if multiple of 11
            if val % 11 == 0:
                candidates.append(val // 11)
            
            for nxt in candidates:
                if nxt == y:
                    return steps + 1
                if 1 <= nxt <= MAX_VAL and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, steps + 1))
        
        # Given the constraints, we should always find a way; theoretically we never reach here.
        return -1