class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        if n == 0:
            return []
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Assign values: 1 for odd nodes, 2 for even nodes
        value = [1 if i % 2 else 2 for i in range(n)]
        
        # Initialize down, max_values, and child_refs
        down = [0] * n
        max_values = [(float('-inf'), float('-inf')) for _ in range(n)]
        child_refs = [(-1, -1) for _ in range(n)]
        
        # Post-order traversal using iterative approach
        stack = [(0, -1, False)]  # (node, parent, visited)
        while stack:
            node, parent, visited = stack.pop()
            if not visited:
                stack.append((node, parent, True))
                # Push children (neighbors except parent) in reverse to maintain order
                for neighbor in reversed(adj[node]):
                    if neighbor != parent:
                        stack.append((neighbor, node, False))
            else:
                max1 = float('-inf')
                child1 = -1
                max2 = float('-inf')
                child2 = -1
                for neighbor in adj[node]:
                    if neighbor != parent:
                        s = value[neighbor] + down[neighbor]
                        if s > max1:
                            max2 = max1
                            child2 = child1
                            max1 = s
                            child1 = neighbor
                        elif s > max2:
                            max2 = s
                            child2 = neighbor
                down[node] = max1 if max1 != float('-inf') else 0
                max_values[node] = (max1, max2)
                child_refs[node] = (child1, child2)
        
        # Initialize result and up arrays
        up = [0] * n
        result = [0] * n
        result[0] = max(down[0], up[0])
        
        from collections import deque
        q = deque()
        q.append((0, -1))  # (current node, parent)
        
        while q:
            u, parent = q.popleft()
            max1, max2 = max_values[u]
            child1, child2 = child_refs[u]
            for v in adj[u]:
                if v != parent:
                    # Determine max_down_without_v
                    if child1 == v:
                        current_max = max(up[u], max2)
                    else:
                        current_max = max(up[u], max1)
                    up[v] = value[u] + current_max
                    result[v] = max(down[v], up[v])
                    q.append((v, u))
        
        return result