from collections import defaultdict
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        # Build tree (adjacency list): node -> list of (child, weight)
        n = len(nums)
        tree = [[] for _ in range(n)]
        for u, v, w in edges:
            tree[u].append((v, w))
            tree[v].append((u, w))
        
        # Global answer: best_length and min number of nodes for that best length.
        best = [0, float('inf')]  # [maxWeight, minNodes]
        
        # cum_list will store cumulative distances along the current DFS ancestry path.
        # At index i, cum_list[i] is the total distance from the start-of-path (the earliest node in the current DFS branch)
        # to the node at index i. 
        cum_list = [0]  # for the first node (if any) the cumulative distance is 0
        
        # lastOccurrence of a value in the current DFS path (map value -> last index along the DFS ancestry)
        lastOccurrence = {}
        
        # DFS: parameters
        # node: current node index
        # parent: parent node in DFS (so that we don’t go back up the tree)
        # window_start: the left boundary index in cum_list (all nodes from window_start to current index have distinct nums)
        # depth: current DFS index (i.e. current length of cum_list minus 1) 
        # curr_cum: current accumulated distance (which is equal to cum_list[depth])
        def dfs(node: int, parent: int, window_start: int, depth: int, curr_cum: int):
            # process current node:
            val = nums[node]
            # if val is seen previously inside our current window, update window_start
            if val in lastOccurrence:
                # We only care if the last occurrence is within current window.
                window_start = max(window_start, lastOccurrence[val] + 1)
            # record previous last occurrence to backtrack later
            prev_occurrence = lastOccurrence.get(val, None)
            lastOccurrence[val] = depth
            
            # Ensure that cum_list has the current cumulative value at position depth.
            # (It should be if we just appended before calling DFS on this node)
            # The window we consider is indices from window_start to depth.
            # Distance of subpath = curr_cum - cum_list[window_start]
            curr_dist = curr_cum - cum_list[window_start]
            # Number of nodes in that subpath:
            node_count = depth - window_start + 1
            # Update global answer:
            if curr_dist > best[0]:
                best[0] = curr_dist
                best[1] = node_count
            elif curr_dist == best[0]:
                best[1] = min(best[1], node_count)
            
            # DFS on children
            for (child, w) in tree[node]:
                if child == parent:
                    continue
                new_cum = curr_cum + w
                # Append new cumulative distance for child.
                cum_list.append(new_cum)
                dfs(child, node, window_start, depth + 1, new_cum)
                cum_list.pop()  # backtrack the cumulative distance
            
            # Backtrack the lastOccurrence for current node's value:
            if prev_occurrence is None:
                del lastOccurrence[val]
            else:
                lastOccurrence[val] = prev_occurrence
            
        # Start DFS from root = 0. 
        # According to problem, tree is rooted at node 0. 
        dfs(0, -1, 0, 0, 0)
        
        return best

# For testing purposes:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    edges = [[0,1,2],[1,2,3],[1,3,5],[1,4,4],[2,5,6]]
    nums = [2,1,2,1,3,1]
    print(sol.longestSpecialPath(edges, nums))  # Expected output [6,2]
    
    # Example 2:
    edges = [[1,0,8]]
    nums = [2,2]
    print(sol.longestSpecialPath(edges, nums))  # Expected output [0,1]