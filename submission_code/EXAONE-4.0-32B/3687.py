import math
from collections import deque
from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        graph = [[] for _ in range(n)]
        for e in edges:
            u, v, w = e
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        depth = [0] * n
        dist = [0] * n
        parent0 = [-1] * n
        q = deque([0])
        depth[0] = 1
        dist[0] = 0
        parent0[0] = -1
        while q:
            u = q.popleft()
            for v, w in graph[u]:
                if v == parent0[u]:
                    continue
                parent0[v] = u
                depth[v] = depth[u] + 1
                dist[v] = dist[u] + w
                q.append(v)
                
        max_log = math.ceil(math.log2(n)) + 1
        parent_table = [[-1] * n for _ in range(max_log)]
        for i in range(n):
            parent_table[0][i] = parent0[i]
        
        for i in range(1, max_log):
            for j in range(n):
                if parent_table[i-1][j] != -1:
                    parent_table[i][j] = parent_table[i-1][parent_table[i-1][j]]
                else:
                    parent_table[i][j] = -1
                    
        last_occ = {}
        global_max_length = 0
        min_nodes_for_max = float('inf')
        stack = [('enter', 0, -1, None)]
        
        while stack:
            state, u, par, in_prev_val = stack.pop()
            if state == 'enter':
                prev_val_saved = last_occ.get(nums[u], None)
                if prev_val_saved is not None:
                    w = prev_val_saved
                    target_depth = depth[w] + 1
                    d_diff = depth[u] - target_depth
                    x = u
                    if d_diff > 0:
                        for i in range(max_log):
                            if d_diff & (1 << i):
                                if x == -1:
                                    break
                                x = parent_table[i][x]
                    if x != -1:
                        length_candidate = dist[u] - dist[x]
                        num_nodes_candidate = depth[u] - depth[x] + 1
                        if length_candidate > global_max_length:
                            global_max_length = length_candidate
                            min_nodes_for_max = num_nodes_candidate
                        elif length_candidate == global_max_length:
                            if num_nodes_candidate < min_nodes_for_max:
                                min_nodes_for_max = num_nodes_candidate
                else:
                    length_candidate = dist[u]
                    num_nodes_candidate = depth[u]
                    if length_candidate > global_max_length:
                        global_max_length = length_candidate
                        min_nodes_for_max = num_nodes_candidate
                    elif length_candidate == global_max_length:
                        if num_nodes_candidate < min_nodes_for_max:
                            min_nodes_for_max = num_nodes_candidate
                
                last_occ[nums[u]] = u
                stack.append(('exit', u, par, prev_val_saved))
                for v, w in graph[u]:
                    if v == par:
                        continue
                    stack.append(('enter', v, u, None))
            else:
                if in_prev_val is not None:
                    last_occ[nums[u]] = in_prev_val
                else:
                    if nums[u] in last_occ:
                        del last_occ[nums[u]]
        
        return [global_max_length, min_nodes_for_max]