from typing import List
import heapq
from collections import defaultdict

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        adj = defaultdict(list)
        degree = [0] * n
        total_weight = 0

        # Build adjacency list, initial degrees, and total weight
        for u, v, w in edges:
            adj[u].append((w, v))
            adj[v].append((w, u))
            degree[u] += 1
            degree[v] += 1
            total_weight += w

        # Store active edges using a set for efficient lookup
        # Use (min(u,v), max(u,v)) as key
        active_edges = set()
        for u, v, w in edges:
            active_edges.add((min(u, v), max(u, v)))

        # Priority queue for each node storing (weight, neighbor) of incident edges
        # Use a list of lists of tuples, then heapify each list
        pq = [[] for _ in range(n)]
        for u in range(n):
            for w, v in adj[u]:
                pq[u].append((w, v))
            heapq.heapify(pq[u])

        # Global priority queue storing (min_weight_incident_edge, node) for nodes with degree > k
        # We prioritize removing the minimum weight edge among *all* violating nodes.
        global_pq = []
        for i in range(n):
            if degree[i] > k:
                # Before adding node i to global_pq, clean up its local pq
                # by removing edges already removed by the other endpoint.
                while pq[i] and (min(i, pq[i][0][1]), max(i, pq[i][0][1])) not in active_edges:
                    heapq.heappop(pq[i])
                
                if pq[i]:
                    # Add the minimum weight edge incident to node i if it violates constraint
                    # pq[i][0][0] is the weight of the smallest active edge incident to i
                    heapq.heappush(global_pq, (pq[i][0][0], i))

        # Process violating nodes using the global priority queue
        while global_pq:
            # Get the node u that has a minimum weight active incident edge among all violating nodes
            min_w, u = heapq.heappop(global_pq)

            # Clean up pq[u]: remove edges already removed by the other endpoint.
            # This is necessary because edges might have been removed since u was added to global_pq.
            while pq[u] and (min(u, pq[u][0][1]), max(u, pq[u][0][1])) not in active_edges:
                heapq.heappop(pq[u])

            # If pq[u] is empty after cleanup, all incident edges are gone for u.
            # This node cannot cause further edge removals.
            if not pq[u]:
                continue
                
            # Check if the extracted entry from global_pq is stale.
            # This happens if the current minimum active edge in pq[u] is heavier
            # than the min_w that was extracted. This indicates the edge corresponding
            # to min_w was already processed (removed by the other endpoint v).
            current_min_w, v = pq[u][0]
            if current_min_w > min_w:
                # The true minimum weight for u is higher now. Push the new minimum back.
                heapq.heappush(global_pq, (current_min_w, u))
                continue

            # Check if node u still violates the degree constraint.
            # Its degree might have dropped to <= k due to edges removed by other nodes.
            if degree[u] <= k:
                # Node u no longer violates, the entry in global_pq is stale in terms of degree constraint.
                continue

            # If we reach here, it means:
            # 1. degree[u] > k (u violates constraint)
            # 2. pq[u] is not empty (u still has active incident edges)
            # 3. (current_min_w, v) = pq[u][0] is the actual minimum weight active edge incident to u
            # 4. current_min_w == min_w (extracted min_w correctly represents the current min edge)
            # This is the minimum weight active edge incident to a violating node. We must remove it.

            w = current_min_w # Weight of the edge (u, v)

            # Remove edge (u, v) by marking it inactive
            edge_key = (min(u, v), max(u, v))
            # This check `if edge_key not in active_edges:` is theoretically not needed here
            # if the logic of cleanup and stale checks is perfect, but adds robustness.
            if edge_key not in active_edges:
                 # This can happen if v also tried to remove this edge and got processed just earlier.
                 # The state has changed since this element was added to global_pq.
                 continue 
                 
            active_edges.remove(edge_key)
            total_weight -= w
            degree[u] -= 1
            degree[v] -= 1

            # Remove (w, v) from pq[u]. It must be the top element after the cleanup loop earlier.
            heapq.heappop(pq[u])
            
            # After removing the edge (u,v), check if u or v still violate the constraint
            # and potentially add them back to global_pq with their new minimum edge.

            # Check node u: if it still violates degree constraint and has active edges
            # Clean up pq[u] to find the next minimum active edge
            while pq[u] and (min(u, pq[u][0][1]), max(u, pq[u][0][1])) not in active_edges:
                heapq.heappop(pq[u])
            if degree[u] > k and pq[u]:
                 # Add u to global_pq with its new minimum active edge weight
                 heapq.heappush(global_pq, (pq[u][0][0], u))
            
            # Check node v: if it now violates or still violates and has active edges
            # Clean up pq[v] to find the next minimum active edge
            while pq[v] and (min(v, pq[v][0][1]), max(v, pq[v][0][1])) not in active_edges:
                heapq.heappop(pq[v])
            if degree[v] > k and pq[v]:
                 # Add v to global_pq with its new minimum active edge weight
                 # Duplicate entries for v in global_pq are fine, the stale checks handle them.
                 heapq.heappush(global_pq, (pq[v][0][0], v))


        return total_weight