import collections

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # We need to find the minimal max edge weight such that:
        # 1. All nodes can reach 0 (i.e., in the reversed graph, 0 can reach all nodes)
        # 2. Each node's out-degree is <= threshold.
        
        # Binary search the minimal possible max edge weight.
        left = 1
        right = max(edge[2] for edge in edges) if edges else 0
        answer = -1
        
        # Pre-process edges: for each node, collect all outgoing edges sorted by weight.
        node_edges = collections.defaultdict(list)
        for a, b, w in edges:
            node_edges[a].append((b, w))
        
        # For each node, sort its edges by weight to facilitate selecting the smallest 'threshold' edges.
        for a in node_edges:
            node_edges[a].sort(key=lambda x: x[1])
        
        def is_possible(max_weight):
            # Build the graph with edges <= max_weight, and each node selects up to threshold edges.
            adj = collections.defaultdict(list)
            for a in node_edges:
                count = 0
                for (b, w) in node_edges[a]:
                    if w > max_weight:
                        break
                    if count < threshold:
                        adj[b].append(a)  # reversed edge for BFS from 0
                        count += 1
            # Now check if 0 can reach all nodes in the reversed graph.
            visited = [False] * n
            queue = collections.deque()
            queue.append(0)
            visited[0] = True
            reached = 1
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if not visited[v]:
                        visited[v] = True
                        reached += 1
                        queue.append(v)
            return reached == n
        
        # Handle case where edges are empty (but n >=2, so impossible unless n is 1, but constraints say n >=2)
        if not edges:
            return -1
        
        # Binary search
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer