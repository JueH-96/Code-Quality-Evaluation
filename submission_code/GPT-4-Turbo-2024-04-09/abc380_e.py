import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    # Initial colors of each cell
    colors = list(range(1, N + 1))
    
    # To keep track of the size of each color
    color_size = [1] * (N + 1)
    
    # To keep track of the parent or representative of each component
    parent = list(range(N + 1))
    
    # Find function with path compression
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # Union by size
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if color_size[rootX] < color_size[rootY]:
                rootX, rootY = rootY, rootX
            parent[rootY] = rootX
            color_size[rootX] += color_size[rootY]
    
    # Output list
    output = []
    
    # Process each query
    for _ in range(Q):
        query_type = int(data[index])
        index += 1
        if query_type == 1:
            x = int(data[index])
            index += 1
            c = int(data[index])
            index += 1
            
            # Find the root of x
            rootX = find(x)
            
            # If the color of the root is already c, continue
            if colors[rootX] == c:
                continue
            
            # Otherwise, change the color of the component of x to c
            original_color = colors[rootX]
            for i in range(1, N + 1):
                if find(i) == rootX:
                    colors[i] = c
            
            # Merge adjacent cells if they have the same new color
            if x > 1 and colors[x - 1] == c:
                union(x, x - 1)
            if x < N and colors[x + 1] == c:
                union(x, x + 1)
        
        elif query_type == 2:
            c = int(data[index])
            index += 1
            
            # Count the number of cells with color c
            count = 0
            for i in range(1, N + 1):
                if find(i) == i and colors[i] == c:
                    count += color_size[i]
            
            output.append(str(count))
    
    # Print all results for query type 2
    print("
".join(output))

if __name__ == "__main__":
    main()