class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        if not edges:
            return [0]
        
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build children list using BFS starting at 0
        children = [[] for _ in range(n)]
        parent = [-1] * n
        from collections import deque
        q = deque([0])
        parent[0] = -1
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and v != parent[u]:
                    parent[v] = u
                    children[u].append(v)
                    q.append(v)
        
        # Compute delta for each node
        delta = [2 if i % 2 == 0 else 1 for i in range(n)]
        
        # Post-order traversal to compute max_down and top_two
        max_down = [0] * n
        top_two = [[] for _ in range(n)]
        
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reverse order to process them in original order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                vals = []
                current_max = 0
                for child in children[node]:
                    val = delta[child] + max_down[child]
                    vals.append((val, child))
                    if val > current_max:
                        current_max = val
                max_down[node] = current_max if vals else 0
                # Sort by descending val and ascending node index (secondary)
                vals.sort(key=lambda x: (-x[0], x[1]))
                top_two[node] = vals.copy()
        
        # Pre-order traversal to compute up_max
        up_max = [-1] * n
        up_max[0] = 0
        q_up = deque([0])
        
        while q_up:
            u = q_up.popleft()
            for v in children[u]:
                if up_max[v] == -1:
                    best_other_val = float('-inf')
                    parent_top_two = top_two[u]
                    for val, child_node in parent_top_two:
                        if child_node != v:
                            best_other_val = val
                            break
                    option1 = best_other_val + delta[u] if best_other_val != float('-inf') else float('-inf')
                    option2 = up_max[u] + delta[u]
                    up_max[v] = max(option1, option2)
                    q_up.append(v)
        
        result = [max(max_down[i], up_max[i]) for i in range(n)]
        return result