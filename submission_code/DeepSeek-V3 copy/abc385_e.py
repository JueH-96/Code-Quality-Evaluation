import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
    
    # Find the center of the tree (either one or two nodes)
    # We can use BFS to find the farthest node, then BFS again to find the diameter
    # and the center is the middle node(s) of the diameter
    def bfs(start):
        visited = [-1] * (N+1)
        q = deque()
        q.append(start)
        visited[start] = 0
        farthest = start
        max_dist = 0
        while q:
            u = q.popleft()
            for v in edges[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    q.append(v)
                    if visited[v] > max_dist:
                        max_dist = visited[v]
                        farthest = v
        return farthest, max_dist
    
    # Find the diameter
    start = 1
    farthest1, _ = bfs(start)
    farthest2, diameter = bfs(farthest1)
    
    # Find the center
    if diameter % 2 == 0:
        center = [farthest1]
        for _ in range(diameter // 2):
            center = [edges[center[0]][0]]
    else:
        center = [farthest1]
        for _ in range(diameter // 2):
            center = [edges[center[0]][0]]
        center = [center[0], edges[center[0]][0]]
    
    # Now, we need to find the root of the snowflake tree
    # The root is the center node with degree x+1, where x is the number of children
    # and each child has y leaves
    # We need to find the root and then count the number of deletions
    
    # Let's try each center node as the root
    min_deletions = float('inf')
    for root in center:
        # BFS to count the number of children and leaves
        visited = set()
        q = deque()
        q.append((root, 0))
        visited.add(root)
        children = []
        while q:
            u, depth = q.popleft()
            if depth == 1:
                children.append(u)
            for v in edges[u]:
                if v not in visited:
                    visited.add(v)
                    q.append((v, depth+1))
        
        # Now, for each child, count the number of leaves
        x = len(children)
        y_counts = []
        for child in children:
            leaves = 0
            q = deque()
            q.append(child)
            visited_child = set()
            visited_child.add(child)
            while q:
                u = q.popleft()
                for v in edges[u]:
                    if v not in visited_child and v != root:
                        visited_child.add(v)
                        if len(edges[v]) == 1:
                            leaves += 1
                        else:
                            q.append(v)
            y_counts.append(leaves)
        
        # Check if all y_counts are the same
        if all(y == y_counts[0] for y in y_counts):
            y = y_counts[0]
            # Calculate the number of deletions
            # The total number of nodes in the snowflake tree is 1 + x + x*y
            # So, the number of deletions is N - (1 + x + x*y)
            total_nodes = 1 + x + x * y
            deletions = N - total_nodes
            if deletions < min_deletions:
                min_deletions = deletions
    
    print(min_deletions)

if __name__ == "__main__":
    main()