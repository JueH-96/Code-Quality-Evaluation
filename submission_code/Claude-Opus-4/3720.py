class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        from collections import defaultdict, deque
        
        def canReachZero(maxWeight):
            # Build graph with edges having weight <= maxWeight
            graph = defaultdict(list)
            for a, b, w in edges:
                if w <= maxWeight:
                    graph[a].append((b, w))
            
            # For each node, we need to find if it can reach node 0
            # while respecting the outdegree constraint
            
            # BFS from node 0 in reverse direction to find which nodes can reach 0
            reverse_graph = defaultdict(list)
            for a, b, w in edges:
                if w <= maxWeight:
                    reverse_graph[b].append((a, w))
            
            # Find all nodes that can reach 0
            can_reach = [False] * n
            can_reach[0] = True
            
            # We need to check if we can build a subgraph where:
            # 1. Each node has outdegree <= threshold
            # 2. All nodes can reach node 0
            
            # Use a more sophisticated approach: for each node, find the best paths to 0
            # considering the outdegree constraint
            
            # Build adjacency list sorted by weight
            adj = defaultdict(list)
            for a, b, w in edges:
                if w <= maxWeight:
                    adj[a].append((w, b))
            
            for node in adj:
                adj[node].sort()
            
            # Check if all nodes can reach 0 with outdegree constraint
            # Use BFS/DFS from each node to check reachability
            for start in range(n):
                if start == 0:
                    continue
                
                # BFS to check if start can reach 0
                visited = [False] * n
                queue = deque([start])
                visited[start] = True
                found = False
                
                while queue and not found:
                    curr = queue.popleft()
                    # Take at most threshold edges from curr
                    count = 0
                    for w, next_node in adj[curr]:
                        if count >= threshold:
                            break
                        if next_node == 0:
                            found = True
                            break
                        if not visited[next_node]:
                            visited[next_node] = True
                            queue.append(next_node)
                        count += 1
                
                if not found:
                    return False
            
            return True
        
        # Binary search on the maximum edge weight
        weights = sorted(set(w for _, _, w in edges))
        
        if not canReachZero(weights[-1]):
            return -1
        
        left, right = 0, len(weights) - 1
        result = weights[-1]
        
        while left <= right:
            mid = (left + right) // 2
            if canReachZero(weights[mid]):
                result = weights[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return result