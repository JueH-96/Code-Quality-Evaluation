from collections import defaultdict, deque

def read_input():
    N, M, S, T = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u, v))
    return N, M, S, T, edges

def build_graph(N, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def solve(N, M, S, T, edges):
    # Build adjacency list representation
    graph = build_graph(N, edges)
    
    # State: (pos_A, pos_B)
    # Initial state: (S, T)
    # Target state: (T, S)
    
    visited = set()
    queue = deque([(S, T, 0)])  # (pos_A, pos_B, moves)
    visited.add((S, T))
    
    while queue:
        pos_A, pos_B, moves = queue.popleft()
        
        # Check if we reached target state
        if pos_A == T and pos_B == S:
            return moves
            
        # Try moving piece A
        for next_A in graph[pos_A]:
            if next_A != pos_B:  # Pieces can't be on same vertex
                if (next_A, pos_B) not in visited:
                    visited.add((next_A, pos_B))
                    queue.append((next_A, pos_B, moves + 1))
        
        # Try moving piece B
        for next_B in graph[pos_B]:
            if next_B != pos_A:  # Pieces can't be on same vertex
                if (pos_A, next_B) not in visited:
                    visited.add((pos_A, next_B))
                    queue.append((pos_A, next_B, moves + 1))
    
    return -1  # No solution found

def main():
    N, M, S, T, edges = read_input()
    result = solve(N, M, S, T, edges)
    print(result)

if __name__ == "__main__":
    main()