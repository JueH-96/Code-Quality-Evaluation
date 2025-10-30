from typing import List
import sys
sys.setrecursionlimit(10**6)

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        # Build the tree as an adjacency list.
        # Each entry graph[u] contains pairs (v, w) meaning there's an edge from u to v with length w.
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
            
        # Global variables to track the best answer.
        # best_length: the maximum sum of edge lengths of a special path.
        # best_nodes: among those paths, the minimum number of nodes.
        best_length = 0
        best_nodes = float('inf')
        
        # last_index will map a node value (nums value) to its most recent index in the current DFS path.
        # The DFS path is maintained implicitly by the "prefix" list and its length.
        last_index = {}
        # "prefix" is a list of cumulative distances along our current path.
        # prefix[i] is the total distance from the first node in the DFS path (starting when we call dfs)
        # up to the node at DFS stack index i.
        # For a single node (path of length 1), the sum is zero (because no edge traveled).
        prefix = [0]
        
        # The DFS function: 
        # u: current node
        # parent: parent's node (to avoid going back)
        # valid_start: the index in the current DFS stack from which the contiguous segment is valid (has unique values)
        def dfs(u: int, parent: int, valid_start: int):
            nonlocal best_length, best_nodes
            
            # Current depth in DFS stack (index for node u).
            cur_depth = len(prefix) - 1
            
            # Current node value.
            cur_val = nums[u]
            # If this value has already appeared in our DFS path,
            # update valid_start so that our unique segment starts after its last occurrence.
            if cur_val in last_index:
                valid_start = max(valid_start, last_index[cur_val] + 1)
            
            # Save the previous occurrence (if any) so we can restore it on backtracking.
            old_occurrence = last_index.get(cur_val)
            last_index[cur_val] = cur_depth
            
            # Calculate the special path that ends at this node.
            # The special path is the contiguous segment from index 'valid_start' to 'cur_depth'
            # Its length is the difference in prefix sums (since prefix sum stores cumulative edge lengths).
            curr_path_length = prefix[cur_depth] - prefix[valid_start]
            # Number of nodes in the path is (cur_depth - valid_start + 1).
            node_count = cur_depth - valid_start + 1
            
            # Update the global best result.
            if curr_path_length > best_length:
                best_length = curr_path_length
                best_nodes = node_count
            elif curr_path_length == best_length and node_count < best_nodes:
                best_nodes = node_count
            
            # Explore the children.
            for nei, w in graph[u]:
                if nei == parent:
                    continue
                # Append the new cumulative distance.
                prefix.append(prefix[-1] + w)
                dfs(nei, u, valid_start)
                prefix.pop()
                
            # Backtrack: restore last_index for current node value.
            if old_occurrence is None:
                del last_index[cur_val]
            else:
                last_index[cur_val] = old_occurrence
                
        dfs(0, -1, 0)
        return [best_length, best_nodes]