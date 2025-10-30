from collections import defaultdict, deque

def color_graph(graph, n):
    """Color the graph using BFS to identify potential odd cycles."""
    colors = [-1] * (n + 1)  # -1 means uncolored
    
    for start in range(1, n + 1):
        if colors[start] == -1:  # If vertex is uncolored
            colors[start] = 0
            queue = deque([start])
            
            while queue:
                vertex = queue.popleft()
                for neighbor in graph[vertex]:
                    if colors[neighbor] == -1:
                        colors[neighbor] = 1 - colors[vertex]  # Assign opposite color
                        queue.append(neighbor)
    
    return colors

def count_valid_moves(graph, n, colors):
    """Count the number of valid moves possible in the game."""
    valid_moves = 0
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # A valid move is adding an edge between vertices that:
            # 1. Don't already have an edge between them
            # 2. Have different colors (to avoid creating an odd cycle)
            if j not in graph[i] and colors[i] != colors[j]:
                valid_moves += 1
    
    return valid_moves

def main():
    N, M = map(int, input().strip().split())
    
    graph = defaultdict(set)
    
    for _ in range(M):
        u, v = map(int, input().strip().split())
        graph[u].add(v)
        graph[v].add(u)
    
    colors = color_graph(graph, N)
    valid_moves = count_valid_moves(graph, N, colors)
    
    # In this impartial game, if the number of valid moves is odd,
    # the first player (Aoki) wins. Otherwise, the second player (Takahashi) wins.
    if valid_moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()