from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If x is less than or equal to y, the best way is to increment until they are equal.
        if x <= y:
            return y - x

        # When x > y, we use a breadth-first search (BFS) to explore the state space.
        # The allowed operations on an integer n are:
        # 1. n - 1  (decrement)
        # 2. n + 1  (increment)
        # 3. n // 11 if n is a multiple of 11 (division)
        # 4. n // 5  if n is a multiple of 5  (division)
        #
        # We also use an upper bound on states to avoid exploring too far.
        # For x > y, it is safe to restrict our search to numbers between 1 and 2*x.
        # (The slight overshoot beyond x is needed so that by incrementing
        #  we might reach a candidate that is divisible by 5 or 11.)
        
        bound = 2 * x  # Upper limit for our search states
        visited = set()
        queue = deque([(x, 0)])  # Each element is a tuple: (current value, operations count)
        visited.add(x)

        while queue:
            current, steps = queue.popleft()
            
            # If we have reached y, return the steps taken so far.
            if current == y:
                return steps

            # Explore the neighbors obtained by decrementing and incrementing.
            for next_val in (current - 1, current + 1):
                if 1 <= next_val <= bound and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # If division by 11 is allowed, explore that neighbor.
            if current % 11 == 0:
                next_val = current // 11
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # If division by 5 is allowed, explore that neighbor.
            if current % 5 == 0:
                next_val = current // 5
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
                    
        # Given the problem constraints, this return will never be reached.
        return -1