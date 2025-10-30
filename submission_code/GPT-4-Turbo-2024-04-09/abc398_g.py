def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict, deque
    
    # Adjacency list of the graph
    graph = defaultdict(list)
    
    index = 2
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        graph[u].append(v)
        graph[v].append(u)
        index += 2
    
    # To check for bipartiteness, we need to color the graph
    color = [-1] * N  # -1 means uncolored, 0 and 1 are the two colors
    
    def bfs(start):
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            current_color = color[node]
            next_color = 1 - current_color
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = next_color
                    queue.append(neighbor)
                elif color[neighbor] == current_color:
                    # This should not happen as the graph is initially bipartite
                    return False
        return True
    
    # Check all components and color them
    for i in range(N):
        if color[i] == -1:
            if not bfs(i):
                print("Aoki")
                return
    
    # Count the number of vertices in each color
    count0 = color.count(0)
    count1 = color.count(1)
    
    # The number of edges that can still be added without creating an odd cycle
    # is the number of edges between different colors minus the edges already present
    max_possible_edges = count0 * count1
    remaining_edges = max_possible_edges - M
    
    # If the remaining edges are odd, Aoki (first player) wins, otherwise Takahashi wins
    if remaining_edges % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()