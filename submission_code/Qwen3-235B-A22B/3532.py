class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        w = [2 if i % 2 == 0 else 1 for i in range(n)]
        
        down = [0] * n
        first_max = [-float('inf')] * n
        second_max = [-float('inf')] * n
        
        # Post-order traversal to compute down, first_max, second_max
        stack = [(0, -1, False)]
        while stack:
            node, parent, visited = stack.pop()
            if not visited:
                stack.append((node, parent, True))
                # Push children in reversed order to process in original order
                for neighbor in reversed(adj[node]):
                    if neighbor != parent:
                        stack.append((neighbor, node, False))
            else:
                # Collect all children
                children = []
                for neighbor in adj[node]:
                    if neighbor != parent:
                        children.append(neighbor)
                # Calculate first_max and second_max
                fm = -float('inf')
                sm = -float('inf')
                for v in children:
                    current_val = w[v] + down[v]
                    if current_val > fm:
                        sm = fm
                        fm = current_val
                    elif current_val > sm:
                        sm = current_val
                first_max[node] = fm
                second_max[node] = sm
                # Compute down value
                down_val = fm if fm != -float('inf') else 0
                down[node] = down_val
        
        # Pre-order traversal to compute up
        up = [0] * n
        stack = [(0, -1)]
        while stack:
            node, parent = stack.pop()
            # Process all children of the current node
            children = []
            for neighbor in adj[node]:
                if neighbor != parent:
                    children.append(neighbor)
            # Iterate in reversed order to push to stack in correct order
            for neighbor in reversed(children):
                v = neighbor
                # Calculate temp_val
                current_v_val = w[v] + down[v]
                if first_max[node] == current_v_val:
                    temp_val = second_max[node]
                else:
                    temp_val = first_max[node]
                # Compute candidate
                candidate = up[node]
                if temp_val > candidate:
                    candidate = temp_val
                # Compute up_val
                up_val = w[node] + candidate
                up[v] = up_val
                # Push child to stack
                stack.append((v, node))
        
        # Compute the answer array
        ans = [0] * n
        for i in range(n):
            ans[i] = max(down[i], up[i])
        
        return ans