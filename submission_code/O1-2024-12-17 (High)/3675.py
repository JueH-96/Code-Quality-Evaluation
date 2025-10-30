from typing import List
import heapq
from collections import deque

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1  # number of nodes in the tree
        
        # Adjacency list: for each node, store (weight, neighbor, edge_id)
        adjacency = [[] for _ in range(n)]
        
        # Degree array
        degree = [0]*n
        
        # Keep track of edge removals
        removed = [False]*(n-1)
        
        # Build adjacency and degree
        for e_id, (u,v,w) in enumerate(edges):
            adjacency[u].append((w,v,e_id))
            adjacency[v].append((w,u,e_id))
            degree[u] += 1
            degree[v] += 1
        
        # Convert adjacency lists to min-heaps based on weights
        for i in range(n):
            heapq.heapify(adjacency[i])
        
        # Queue of nodes over the limit
        in_queue = [False]*n
        queue = deque()
        for i in range(n):
            if degree[i] > k:
                queue.append(i)
                in_queue[i] = True

        # Function to skip stale (already removed) edges at the top of the heap
        def get_valid_edge(i):
            # Pop edges that were already removed
            while adjacency[i] and removed[adjacency[i][0][2]]:
                heapq.heappop(adjacency[i])
            return adjacency[i][0] if adjacency[i] else None

        # Process all nodes that are over the limit
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            
            # While this node's degree is over k, remove the smallest-weight edge
            while degree[node] > k:
                top = get_valid_edge(node)
                if not top:  # No more edges to remove (should not happen in a valid tree if deg>k)
                    break
                w, nbr, e_id = top
                # Mark edge as removed
                removed[e_id] = True
                # Reduce degrees
                degree[node] -= 1
                degree[nbr] -= 1
                # Pop from heap
                heapq.heappop(adjacency[node])
                
                # If neighbor is still above limit, re-queue if not already in queue
                if degree[nbr] > k and not in_queue[nbr]:
                    queue.append(nbr)
                    in_queue[nbr] = True
        
        # Sum weights of remaining edges
        total = 0
        for e_id, (u, v, w) in enumerate(edges):
            if not removed[e_id]:
                total += w
        
        return total