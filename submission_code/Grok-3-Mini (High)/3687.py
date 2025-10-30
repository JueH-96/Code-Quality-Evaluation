import sys
from typing import List

sys.setrecursionlimit(100005)  # Increase recursion depth if necessary

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        adj = [[] for _ in range(n)]
        for u, v, length in edges:
            adj[u].append((v, length))
            adj[v].append((u, length))

        def longest_from_node(node, parent, used):
            if nums[node] in used:
                return 0, 0  # Cannot include this node
            used.add(nums[node])
            max_len = 0
            min_nodes_for_max = 1  # For the node itself

            for neigh, length in adj[node]:
                if neigh != parent and nums[neigh] not in used:
                    child_len, child_nodes = longest_from_node(neigh, node, used)
                    total_len = length + child_len
                    total_nodes = 1 + child_nodes
                    if total_len > max_len:
                        max_len = total_len
                        min_nodes_for_max = total_nodes
                    elif total_len == max_len:
                        min_nodes_for_max = min(min_nodes_for_max, total_nodes)

            used.remove(nums[node])
            return max_len, min_nodes_for_max

        global_max_len = 0
        global_min_nodes = float('inf')
        used_set = set()

        for s in range(n):
            len_s, nodes_s = longest_from_node(s, -1, used_set)
            if len_s > global_max_len:
                global_max_len = len_s
                global_min_nodes = nodes_s
            elif len_s == global_max_len:
                if nodes_s < global_min_nodes:
                    global_min_nodes = nodes_s

        return [global_max_len, global_min_nodes]