# YOUR CODE HERE
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

N, Q = map(int, input().split())

# Initialize colors - cell i has color i (0-indexed internally)
colors = list(range(N))

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: Repaint connected component
        x, c = query[1] - 1, query[2] - 1  # Convert to 0-indexed
        
        # Find all cells in the connected component
        target_color = colors[x]
        component = []
        
        # Use BFS to find connected component
        visited = [False] * N
        queue = [x]
        visited[x] = True
        
        while queue:
            curr = queue.pop(0)
            component.append(curr)
            
            # Check left neighbor
            if curr > 0 and not visited[curr-1] and colors[curr-1] == target_color:
                visited[curr-1] = True
                queue.append(curr-1)
            
            # Check right neighbor
            if curr < N-1 and not visited[curr+1] and colors[curr+1] == target_color:
                visited[curr+1] = True
                queue.append(curr+1)
        
        # Repaint all cells in component
        for cell in component:
            colors[cell] = c
            
    else:
        # Type 2: Count cells with color c
        c = query[1] - 1  # Convert to 0-indexed
        count = sum(1 for color in colors if color == c)
        print(count)