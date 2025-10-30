class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n1 = len(edges1) + 1
        n2 = len(edges2) + 1
        
        adj_list1 = self.build_adjacency_list(edges1, n1)
        adj_list2 = self.build_adjacency_list(edges2, n2)
        
        # Find the diameter and its endpoints for the first tree
        end1_1, _, _ = self.find_farthest_node(0, adj_list1)
        end1_2, d1, _ = self.find_farthest_node(end1_1, adj_list1)
        
        # Find the diameter and its endpoints for the second tree
        end2_1, _, _ = self.find_farthest_node(0, adj_list2)
        end2_2, d2, _ = self.find_farthest_node(end2_1, adj_list2)
        
        # Compute the eccentricity of each node
        eccentricities1 = self.compute_eccentricities(adj_list1, end1_1, end1_2)
        eccentricities2 = self.compute_eccentricities(adj_list2, end2_1, end2_2)
        
        # Find the node with the minimum eccentricity in each tree
        min_eccentricity1 = min(eccentricities1)
        min_eccentricity2 = min(eccentricities2)
        
        # Connect the centers of the two trees and compute the resulting diameter
        return max(d1, d2, min_eccentricity1 + 1 + min_eccentricity2)
    
    def build_adjacency_list(self, edges, n):
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        return adj_list

    def find_farthest_node(self, start, adj_list):
        n = len(adj_list)
        distances = [-1] * n
        distances[start] = 0
        parent = [-1] * n
        
        queue = [start]
        farthest_node = start
        max_distance = 0
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                    if distances[neighbor] > max_distance:
                        max_distance = distances[neighbor]
                        farthest_node = neighbor
        
        # Reconstruct the path from start to farthest_node
        path = [farthest_node]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path = path[::-1]
        
        return farthest_node, max_distance, path

    def compute_eccentricities(self, adj_list, end1, end2):
        n = len(adj_list)
        
        # Compute distances from end1 to all nodes
        distances1 = [-1] * n
        distances1[end1] = 0
        queue = [end1]
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if distances1[neighbor] == -1:
                    distances1[neighbor] = distances1[node] + 1
                    queue.append(neighbor)
        
        # Compute distances from end2 to all nodes
        distances2 = [-1] * n
        distances2[end2] = 0
        queue = [end2]
        while queue:
            node = queue.pop(0)
            for neighbor in adj_list[node]:
                if distances2[neighbor] == -1:
                    distances2[neighbor] = distances2[node] + 1
                    queue.append(neighbor)
        
        # For each node, the eccentricity is the maximum distance to either end1 or end2
        eccentricities = [max(distances1[i], distances2[i]) for i in range(n)]
        
        return eccentricities