from collections import deque

def solve():
    # Read input
    N, M, S, T = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS to find minimum moves
    # State: (position of A, position of B)
    start = (S, T)
    target = (T, S)
    
    if start == target:
        print(0)
        return
    
    queue = deque([start])
    visited = {start}
    distance = {start: 0}
    
    while queue:
        pos_a, pos_b = queue.popleft()
        curr_dist = distance[(pos_a, pos_b)]
        
        # Try moving piece A
        for next_a in adj[pos_a]:
            if next_a != pos_b:  # Cannot move to where B is
                next_state = (next_a, pos_b)
                if next_state not in visited:
                    visited.add(next_state)
                    distance[next_state] = curr_dist + 1
                    queue.append(next_state)
                    
                    if next_state == target:
                        print(curr_dist + 1)
                        return
        
        # Try moving piece B
        for next_b in adj[pos_b]:
            if next_b != pos_a:  # Cannot move to where A is
                next_state = (pos_a, next_b)
                if next_state not in visited:
                    visited.add(next_state)
                    distance[next_state] = curr_dist + 1
                    queue.append(next_state)
                    
                    if next_state == target:
                        print(curr_dist + 1)
                        return
    
    # If we couldn't reach the target state
    print(-1)

solve()