from collections import deque, defaultdict

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    S = int(data[index])
    index += 1
    T = int(data[index])
    index += 1
    
    adjacency_list = defaultdict(list)
    
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    # BFS setup
    queue = deque([(S, T, 0)])  # (position of A, position of B, number of moves)
    visited = set()
    visited.add((S, T))
    
    while queue:
        pos_A, pos_B, moves = queue.popleft()
        
        # Check if we reached the goal state
        if pos_A == T and pos_B == S:
            print(moves)
            return
        
        # Move A
        for neighbor_A in adjacency_list[pos_A]:
            if neighbor_A != pos_B:  # Ensure A and B don't end up on the same vertex
                new_state = (neighbor_A, pos_B)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((neighbor_A, pos_B, moves + 1))
        
        # Move B
        for neighbor_B in adjacency_list[pos_B]:
            if neighbor_B != pos_A:  # Ensure A and B don't end up on the same vertex
                new_state = (pos_A, neighbor_B)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((pos_A, neighbor_B, moves + 1))
    
    # If we exhaust the queue without finding the goal state
    print(-1)