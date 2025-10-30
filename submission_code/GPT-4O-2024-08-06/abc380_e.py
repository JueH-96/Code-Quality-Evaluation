# YOUR CODE HERE
import sys
input = sys.stdin.read

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def main():
    data = input().split()
    N = int(data[0])
    Q = int(data[1])
    
    queries = data[2:]
    
    # Initialize union-find structure
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    # Initial color of each cell is its index
    color = list(range(N + 1))
    
    # Initialize color count
    color_count = {i: 1 for i in range(1, N + 1)}
    
    output = []
    
    index = 0
    for _ in range(Q):
        query_type = int(queries[index])
        if query_type == 1:
            x = int(queries[index + 1])
            c = int(queries[index + 2])
            index += 3
            
            # Find the root of the component containing x
            root = find(parent, x)
            original_color = color[root]
            
            if original_color != c:
                # Repaint the component
                stack = [root]
                visited = set(stack)
                
                while stack:
                    node = stack.pop()
                    color_count[color[node]] -= 1
                    color[node] = c
                    color_count[c] = color_count.get(c, 0) + 1
                    
                    # Check adjacent nodes
                    for neighbor in (node - 1, node + 1):
                        if 1 <= neighbor <= N and neighbor not in visited:
                            if color[find(parent, neighbor)] == original_color:
                                union(parent, rank, node, neighbor)
                                stack.append(neighbor)
                                visited.add(neighbor)
        
        elif query_type == 2:
            c = int(queries[index + 1])
            index += 2
            output.append(str(color_count.get(c, 0)))
    
    sys.stdout.write("
".join(output) + "
")

main()