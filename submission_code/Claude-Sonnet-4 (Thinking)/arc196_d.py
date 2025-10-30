class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

def solve_query(people, N):
    uf = UnionFind(N)
    
    # Enforce equality constraints P_{S-1} = P_{T-1}
    for S, T in people:
        uf.union(S - 1, T - 1)
    
    # Build inequality graph between connected components
    from collections import defaultdict
    graph = defaultdict(set)
    
    for S, T in people:
        start_idx = S - 1
        if S < T:
            # Intermediate positions need P_k > P_{S-1}
            for idx in range(S, T - 1):
                comp_start = uf.find(start_idx)
                comp_intermediate = uf.find(idx)
                if comp_start == comp_intermediate:
                    return False  # Direct conflict
                graph[comp_start].add(comp_intermediate)
        else:
            # Intermediate positions need P_{S-1} > P_k
            for idx in range(T, S - 1):
                comp_start = uf.find(start_idx)
                comp_intermediate = uf.find(idx)
                if comp_start == comp_intermediate:
                    return False  # Direct conflict
                graph[comp_intermediate].add(comp_start)
    
    # Check for cycles in the directed graph
    def has_cycle():
        color = {}  # 0 = white, 1 = gray, 2 = black
        
        def dfs(node):
            if node in color:
                return color[node] == 1  # Gray means cycle
            color[node] = 1  # Gray
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            color[node] = 2  # Black
            return False
        
        for node in graph:
            if node not in color and dfs(node):
                return True
        return False
    
    return not has_cycle()

# Read input
N, M, Q = map(int, input().split())
people = []
for _ in range(M):
    S, T = map(int, input().split())
    people.append((S, T))

for _ in range(Q):
    L, R = map(int, input().split())
    query_people = people[L-1:R]
    if solve_query(query_people, N):
        print("Yes")
    else:
        print("No")