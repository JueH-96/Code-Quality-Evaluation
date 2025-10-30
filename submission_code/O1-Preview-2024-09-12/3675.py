class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        import heapq
        n = len(edges) + 1
        deg = [0] * n
        total_weight = 0
        edge_list = []
        for idx, (u, v, w) in enumerate(edges):
            deg[u] +=1
            deg[v] +=1
            total_weight += w
            edge_list.append( (w, u, v, idx) )
        
        removed = [False] * len(edges)
        heapq.heapify(edge_list)  # Min-heap based on weight
        
        while edge_list:
            w, u, v, idx = heapq.heappop(edge_list)
            if deg[u] > k or deg[v] > k:
                # Remove edge
                deg[u] -=1
                deg[v] -=1
                total_weight -= w
                removed[idx] = True
            else:
                # No need to remove further
                continue
        return total_weight