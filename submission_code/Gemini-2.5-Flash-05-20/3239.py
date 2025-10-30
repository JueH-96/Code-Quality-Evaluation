import collections

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If x is already equal to y, 0 operations are needed.
        if x == y:
            return 0

        # Queue for BFS, storing tuples of (current_value, steps).
        queue = collections.deque([(x, 0)])
        
        # Set to keep track of visited numbers to avoid cycles and redundant computations.
        visited = {x}
        
        # Determine a safe upper bound for numbers to explore.
        # Incrementing 'x' beyond max(initial_x, y) might be useful if it leads to
        # a number that is a multiple of 5 or 11, allowing a significant reduction.
        # The largest 'k' needed to reach the next multiple of 5 or 11 is at most 10 (e.g., from X to X+10 for next multiple of 11).
        # So, if initial x or y is 10000, we might need to explore up to 10000 + 10 = 10010.
        # A slightly larger buffer like 15 ensures robustness.
        # For example, if current_val is 1, and y is 10000, we need to reach 10000.
        # The upper limit of values in the search space can be `max(x, y) + some_small_constant`.
        # Let's use `max(x, y) + 15` as a reasonable and safe upper bound.
        MAX_EXPLORE_VALUE = max(x, y) + 15 

        while queue:
            current_val, steps = queue.popleft()

            # If the current value is y, we've found the shortest path.
            if current_val == y:
                return steps

            # Explore possible next states:
            
            # 1. Divide x by 11 if x is a multiple of 11.
            if current_val % 11 == 0:
                next_val = current_val // 11
                # Ensure the number stays positive (>= 1) and hasn't been visited.
                if next_val >= 1 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # 2. Divide x by 5 if x is a multiple of 5.
            if current_val % 5 == 0:
                next_val = current_val // 5
                # Ensure the number stays positive (>= 1) and hasn't been visited.
                if next_val >= 1 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, steps + 1))
            
            # 3. Decrement x by 1.
            next_val = current_val - 1
            # Ensure the number stays positive (>= 1) and hasn't been visited.
            if next_val >= 1 and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
            
            # 4. Increment x by 1.
            next_val = current_val + 1
            # Only explore if next_val is within our defined search space limit
            # and hasn't been visited. This prevents unbounded growth and
            # exploring numbers that are clearly suboptimal to reach.
            if next_val <= MAX_EXPLORE_VALUE and next_val not in visited:
                visited.add(next_val)
                queue.append((next_val, steps + 1))
        
        # This part should ideally not be reached if y is always reachable from x.
        # Given the problem constraints and operations, it implies reachability.
        # It's included for completeness, though typically a valid path always exists.
        return -1