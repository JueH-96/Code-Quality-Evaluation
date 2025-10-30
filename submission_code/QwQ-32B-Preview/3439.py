from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_adjacency_list(edges, n):
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            return adj
        
        def find_centers(adj, n):
            if n == 1:
                return [0]
            degrees = [len(adj[i]) for i in range(n)]
            leaves = deque([i for i in range(n) if degrees[i] == 1])
            remaining = n
            while remaining > 2:
                leaf_count = len(leaves)
                remaining -= leaf_count
                for _ in range(leaf_count):
                    leaf = leaves.popleft()
                    for neighbor in adj[leaf]:
                        degrees[neighbor] -= 1
                        if degrees[neighbor] == 1:
                            leaves.append(neighbor)
            return list(leaves)
        
        def compute_eccentricity(adj, center, n):
            visited = [False] * n
            queue = deque([center])
            visited[center] = True
            distance = -1
            max_distance = 0
            while queue:
                level_size = len(queue)
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                distance += 1
                max_distance = distance
            return max_distance
        
        def compute_diameter(adj, n):
            # Find the farthest node from an arbitrary starting node
            start = 0
            farthest, _ = bfs farthest(adj, start)
            # Find the farthest node from the farthest node found
            diameter_end, diameter = bfs_farthest(adj, farthest)
            return diameter
        
        def bfs_farthest(adj, start):
            visited = [False] * len(adj)
            queue = deque([start])
            visited[start] = True
            distance = 0
            farthest = start
            while queue:
                level_size = len(queue)
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                            farthest = neighbor
                distance += 1
            return farthest, distance - 1  # Subtract 1 because distance increments after processing a level
        
        # Step 1: Build adjacency lists for both trees
        n = len(edges1) + 1
        m = len(edges2) + 1
        adj1 = build_adjacency_list(edges1, n)
        adj2 = build_adjacency_list(edges2, m)
        
        # Step 2: Find centers of both trees
        centers1 = find_centers(adj1, n)
        centers2 = find_centers(adj2, m)
        
        # Step 3: Compute diameters of both trees
        D1 = compute_diameter(adj1, n)
        D2 = compute_diameter(adj2, m)
        
        # Step 4: Compute eccentricities of centers
        min_sum = float('inf')
        for center1 in centers1:
            ecc1 = compute_eccentricity(adj1, center1, n)
            for center2 in centers2:
                ecc2 = compute_eccentricity(adj2, center2, m)
                current_sum = ecc1 + ecc2 + 1
                if current_sum < min_sum:
                    min_sum = current_sum
        
        # Step 5: The minimum diameter is the maximum of D1, D2, and the minimized sum
        return max(D1, D2, min_sum)