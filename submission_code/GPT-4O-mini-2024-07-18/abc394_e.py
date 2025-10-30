def shortest_palindrome_paths(N, edges):
    from collections import deque, defaultdict
    
    # Initialize the result matrix with -1
    result = [[-1] * N for _ in range(N)]
    
    # Function to perform BFS and find shortest palindrome paths
    def bfs(start):
        queue = deque([(start, "", 0)])  # (current_node, current_path, current_length)
        visited = defaultdict(lambda: float('inf'))  # Track the shortest length to each node with a specific path
        visited[(start, "")] = 0
        
        while queue:
            current_node, current_path, current_length = queue.popleft()
            
            # Check if current path is a palindrome
            if current_path == current_path[::-1]:
                result[start][current_node] = current_length
            
            # Explore neighbors
            for neighbor in range(N):
                edge_label = edges[current_node][neighbor]
                if edge_label != '-':
                    new_path = current_path + edge_label
                    new_length = current_length + 1
                    
                    # Only enqueue if we found a shorter path to (neighbor, new_path)
                    if new_length < visited[(neighbor, new_path)]:
                        visited[(neighbor, new_path)] = new_length
                        queue.append((neighbor, new_path, new_length))
    
    # Run BFS for each node
    for i in range(N):
        bfs(i)
    
    # Fill in the diagonal with 0s (distance from a node to itself)
    for i in range(N):
        result[i][i] = 0
    
    return result

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    edges = [list(data[i + 1]) for i in range(N)]
    
    result = shortest_palindrome_paths(N, edges)
    
    for row in result:
        print(" ".join(str(x) for x in row))

if __name__ == "__main__":
    main()