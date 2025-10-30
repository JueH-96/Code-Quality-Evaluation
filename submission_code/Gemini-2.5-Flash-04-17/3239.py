import collections

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # If x is already equal to y, no operations needed.
        if x == y:
            return 0

        # Use BFS to find the shortest path
        # Queue stores tuples of (current_value, operations_count)
        queue = collections.deque([(x, 0)])
        
        # Set to keep track of visited values to avoid cycles and redundant computations.
        # In an unweighted graph, the first time a node is visited in BFS is via the shortest path.
        # This set effectively stores the minimum cost to reach a state implicitly.
        visited = {x} 

        while queue:
            curr, ops = queue.popleft()

            # Check if we reached the target
            if curr == y:
                return ops

            # Explore possible operations
            # 1. Divide by 11
            # Check if curr is a multiple of 11 and the result is non-negative.
            # Since the target y is positive (>= 1), negative intermediate values are not useful.
            if curr % 11 == 0:
                next_val = curr // 11
                if next_val >= 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))

            # 2. Divide by 5
            # Check if curr is a multiple of 5 and the result is non-negative.
            if curr % 5 == 0:
                next_val = curr // 5
                if next_val >= 0 and next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))

            # 3. Decrement by 1
            next_val = curr - 1
            # We allow visiting 0, as it might be necessary to reach 1 (e.g., 0 + 1 = 1).
            # Negative values are not useful for reaching a positive target y.
            if next_val >= 0:
                if next_val not in visited:
                    visited.add(next_val)
                    queue.append((next_val, ops + 1))

            # 4. Increment by 1
            next_val = curr + 1
            # For BFS on an unweighted graph, we explore neighbors level by level.
            # The 'visited' set prevents revisiting states already processed or enqueued.
            # For the given constraints (x, y <= 10^4), the state space explored
            # by BFS before finding the target y is expected to be manageable due to
            # the presence of division operations which reduce the number significantly.
            # We do not impose an arbitrary upper bound on `next_val` for the increment operation,
            # relying on the nature of the problem and BFS properties.
            if next_val not in visited:
                 visited.add(next_val)
                 queue.append((next_val, ops + 1))

        # The loop will always find a path because decrement/increment allows reaching any non-negative integer from any non-negative integer.
        # Since x and y are positive, decrementing can reach 0, and incrementing from 0 can reach any positive y.
        # Thus, y is always reachable from x. The BFS is guaranteed to find y.
        # This return statement should logically be unreachable given the problem constraints.
        # It's included for completeness but indicates an issue if reached during execution.
        return -1 # Should theoretically not be reached.