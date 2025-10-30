import sys
from collections import deque, defaultdict

def solve():
    """
    Reads the problem input, solves it, and prints the output.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Bipartite graph adjacency list
    # Keys: 1..M for numbers, M+1..M+N for sets
    adj = defaultdict(list)
    
    # Store sets containing 1 (start points) and M (end points)
    sets_with_1 = []
    sets_with_m = set()

    # We use 1-based indexing for sets (1 to N) for clarity
    for i in range(1, N + 1):
        # A_i is read but not used directly
        sys.stdin.readline()
        elements_str = sys.stdin.readline().split()
        if not elements_str:
            continue
        
        elements = [int(x) for x in elements_str]
        
        set_node = M + i
        
        has_1 = False
        has_m = False
        
        # Build graph and check for presence of 1 and M
        for k in elements:
            if k == 1:
                has_1 = True
            if k == M:
                has_m = True
            adj[k].append(set_node)
            adj[set_node].append(k)

        # Handle the 0-operation case
        if has_1 and has_m:
            print(0)
            return
        
        if has_1:
            sets_with_1.append(i)
        if has_m:
            sets_with_m.add(i)

    # If no set contains 1 or no set contains M, connection is impossible
    if not sets_with_1 or not sets_with_m:
        print(-1)
        return

    # BFS to find the shortest path in terms of set merges
    # dist[i] = min operations to connect set i to a set with 1
    dist = [-1] * (N + 1)
    # visited_nums prevents using a number as a bridge more than once
    visited_nums = [False] * (M + 1)
    
    q = deque()

    # Initialize queue with all sets containing 1 (the sources)
    for start_set_idx in sets_with_1:
        if dist[start_set_idx] == -1:
            dist[start_set_idx] = 0
            q.append(start_set_idx)

    # Main BFS loop
    while q:
        set_idx = q.popleft() # 1-indexed set index
        
        # If the current set contains M, we found a shortest path
        if set_idx in sets_with_m:
            print(dist[set_idx])
            return

        set_node = M + set_idx
        
        # Explore neighbors via common numbers
        for num_node in adj[set_node]:
            if visited_nums[num_node]:
                continue
            visited_nums[num_node] = True
            
            # Find all other sets connected via this number
            for next_set_node in adj[num_node]:
                next_set_idx = next_set_node - M
                if dist[next_set_idx] == -1:
                    dist[next_set_idx] = dist[set_idx] + 1
                    q.append(next_set_idx)
    
    # If the queue empties and no target set was reached, it's impossible
    print(-1)

if __name__ == "__main__":
    solve()