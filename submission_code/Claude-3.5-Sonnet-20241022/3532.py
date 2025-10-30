class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # Build adjacency list representation of tree
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def bfs(start: int) -> int:
            marked = set()
            q = [(start, 0)]  # (node, time)
            marked.add(start)
            max_time = 0
            
            while q:
                new_q = []
                for node, time in q:
                    for neighbor in adj[node]:
                        if neighbor not in marked:
                            # For odd numbered nodes
                            if neighbor % 2 == 1:
                                # Check if any neighbor was marked in previous time step
                                if time + 1 not in marked:
                                    new_q.append((neighbor, time + 1))
                                    marked.add(neighbor)
                                    max_time = max(max_time, time + 1)
                            # For even numbered nodes        
                            else:
                                # Check if any neighbor was marked 2 time steps ago
                                if time + 2 not in marked:
                                    new_q.append((neighbor, time + 2))
                                    marked.add(neighbor)
                                    max_time = max(max_time, time + 2)
                q = new_q
                
            return max_time if len(marked) == n else 0
            
        result = []
        # Try marking each node at time 0
        for i in range(n):
            result.append(bfs(i))
            
        return result