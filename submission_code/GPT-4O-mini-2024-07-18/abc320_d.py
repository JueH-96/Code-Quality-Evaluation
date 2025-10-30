from collections import defaultdict, deque
import sys

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Adjacency list for the graph
    graph = defaultdict(list)
    # To store the relative positions
    positions = defaultdict(list)
    
    # Read the M pieces of information
    for i in range(1, M + 1):
        A, B, X, Y = map(int, data[i].split())
        graph[A].append(B)
        graph[B].append(A)
        positions[A].append((B, X, Y))
        positions[B].append((A, -X, -Y))
    
    # To store the coordinates of each person
    coords = {}
    coords[1] = (0, 0)  # Person 1 is at the origin
    
    # BFS to determine coordinates
    queue = deque([1])
    visited = set([1])
    
    while queue:
        current = queue.popleft()
        current_x, current_y = coords[current]
        
        for neighbor, dx, dy in positions[current]:
            if neighbor not in visited:
                # Calculate the coordinates of the neighbor
                coords[neighbor] = (current_x + dx, current_y + dy)
                visited.add(neighbor)
                queue.append(neighbor)
            else:
                # If already visited, check for consistency
                neighbor_x, neighbor_y = coords[neighbor]
                expected_x = current_x + dx
                expected_y = current_y + dy
                if (neighbor_x, neighbor_y) != (expected_x, expected_y):
                    coords[neighbor] = "undecidable"
    
    # Prepare output
    output = []
    for i in range(1, N + 1):
        if i in coords:
            if coords[i] == "undecidable":
                output.append("undecidable")
            else:
                output.append(f"{coords[i][0]} {coords[i][1]}")
        else:
            output.append("undecidable")
    
    print("
".join(output))

if __name__ == "__main__":
    main()