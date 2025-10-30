from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        # We'll use BFS where each node is a value of x and cost is the number of operations.
        # Since x and y are positive and constrained upto 10^4, we set a safe upper bound.
        # Note: Even if we can increment or decrement, division operations only reduce the number.
        # So we consider boundaries similar to [0, max_val]. For safety, we use an upper bound a bit
        # larger than max(x, y).
        #
        # We can compute an upper bound as max(x, y) + abs(x - y) + 10.
        # Also, the lower bound can be 0 because x is positive (and decrement could lead to 0).
        
        if x == y:
            return 0
        
        max_bound = max(x, y) + abs(x - y) + 10
        
        # BFS queue will store tuples: (current value, steps)
        queue = deque([(x, 0)])
        visited = set([x])
        
        while queue:
            curr, steps = queue.popleft()
            
            # Try possible operations:
            candidates = []
            
            # Operation 1: Divide x by 11 if divisible
            if curr % 11 == 0:
                candidates.append(curr // 11)
            # Operation 2: Divide x by 5 if divisible
            if curr % 5 == 0:
                candidates.append(curr // 5)
            # Operation 3: Decrement by 1 if possible (ensure non-negative)
            if curr - 1 >= 0:
                candidates.append(curr - 1)
            # Operation 4: Increment by 1 (limited by bound)
            if curr + 1 <= max_bound:
                candidates.append(curr + 1)
                
            for new_val in candidates:
                if new_val == y:
                    return steps + 1
                if new_val not in visited:
                    visited.add(new_val)
                    queue.append((new_val, steps + 1))
        
        # If we haven't found y (should not happen given the problem constraints),
        # return -1 as error.
        return -1

# You can perform some tests as follows:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumOperationsToMakeEqual(26, 1))  # Expected output: 3
    print(sol.minimumOperationsToMakeEqual(54, 2))  # Expected output: 4
    print(sol.minimumOperationsToMakeEqual(25, 30)) # Expected output: 5