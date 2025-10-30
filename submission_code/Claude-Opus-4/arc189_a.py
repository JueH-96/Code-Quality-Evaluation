def solve():
    MOD = 998244353
    
    N = int(input())
    A = list(map(int, input().split()))
    
    # Initial configuration
    initial = [i % 2 for i in range(1, N + 1)]
    
    # Check if target is reachable
    # We can only change 0s to 1s or 1s to 0s through operations
    
    # Dynamic programming approach
    # dp[mask] = number of ways to reach configuration represented by mask
    # But N can be up to 2*10^5, so we need a different approach
    
    # Key insight: We need to track intervals and their formation order
    # Let's use interval DP
    
    # dp[l][r][v] = number of ways to make interval [l,r] all have value v
    # But this might be too memory intensive
    
    # Alternative: Since operations must satisfy specific conditions,
    # we can recursively count valid operation sequences
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def count_ways(config):
        # Convert list to tuple for hashing
        if config == tuple(A):
            return 1
        
        total = 0
        # Try all possible operations
        for l in range(N):
            for r in range(l + 2, N):
                # Check if operation is valid
                if config[l] == config[r]:
                    # Check all intermediate cells
                    valid = True
                    for i in range(l + 1, r):
                        if config[i] == config[l]:
                            valid = False
                            break
                    
                    if valid:
                        # Apply operation
                        new_config = list(config)
                        for i in range(l + 1, r):
                            new_config[i] = config[l]
                        
                        total = (total + count_ways(tuple(new_config))) % MOD
        
        return total
    
    # Start from initial configuration
    result = count_ways(tuple(initial))
    print(result)

# This approach might be too slow for large N
# Let's think of a more efficient solution

def solve_efficient():
    MOD = 998244353
    
    N = int(input())
    A = list(map(int, input().split()))
    
    # Initial configuration
    initial = [(i + 1) % 2 for i in range(N)]
    
    # We need to count the number of operation sequences
    # Key insight: Each operation creates a segment of identical values
    # We need to work backwards from the target to initial
    
    # Let's think differently: simulate all possible sequences
    from collections import deque
    
    # State: (current configuration, number of ways to reach it)
    visited = {}
    queue = deque()
    
    initial_tuple = tuple(initial)
    target_tuple = tuple(A)
    
    queue.append(initial_tuple)
    visited[initial_tuple] = 1
    
    while queue:
        current = queue.popleft()
        current_ways = visited[current]
        
        # Try all possible operations
        for l in range(N):
            for r in range(l + 2, N):
                # Check if operation is valid
                if current[l] == current[r]:
                    # Check all intermediate cells
                    valid = True
                    for i in range(l + 1, r):
                        if current[i] == current[l]:
                            valid = False
                            break
                    
                    if valid:
                        # Apply operation
                        new_config = list(current)
                        for i in range(l + 1, r):
                            new_config[i] = current[l]
                        new_tuple = tuple(new_config)
                        
                        if new_tuple not in visited:
                            visited[new_tuple] = 0
                            queue.append(new_tuple)
                        
                        visited[new_tuple] = (visited[new_tuple] + current_ways) % MOD
    
    print(visited.get(target_tuple, 0))

solve_efficient()