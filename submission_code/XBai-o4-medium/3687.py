from typing import List

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        def contains(n: int, x: int, best_child: List[int], nums: List[int]) -> bool:
            while n != -1:
                if nums[n] == x:
                    return True
                n = best_child[n]
            return False

        n = len(edges) + 1
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        
        # Build children structure with root at 0
        children = [[] for _ in range(n)]
        visited = [False] * n
        stack = [0]
        visited[0] = True
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    children[u].append((v, w))
                    stack.append(v)
        
        # Prepare for post-order traversal
        max_len = [0] * n
        count = [0] * n
        best_child = [-1] * n
        global_max_length = 0
        global_min_count = float('inf')
        
        stack = [(0, False)]
        while stack:
            node, is_processed = stack.pop()
            if not is_processed:
                stack.append((node, True))
                # Push children in reversed order to process them in order
                for v, w in reversed(children[node]):
                    stack.append((v, False))
            else:
                current_value = nums[node]
                current_max = 0
                current_cnt = 1
                best_child_node = -1
                for v, w in children[node]:
                    if not contains(v, current_value, best_child, nums):
                        candidate_len = w + max_len[v]
                        candidate_cnt = 1 + count[v]
                        if candidate_len > current_max:
                            current_max = candidate_len
                            current_cnt = candidate_cnt
                            best_child_node = v
                        elif candidate_len == current_max:
                            if candidate_cnt < current_cnt:
                                current_cnt = candidate_cnt
                                best_child_node = v
                max_len[node] = current_max
                count[node] = current_cnt
                best_child[node] = best_child_node
                # Update global results
                if current_max > global_max_length:
                    global_max_length = current_max
                    global_min_count = current_cnt
                elif current_max == global_max_length:
                    if current_cnt < global_min_count:
                        global_min_count = current_cnt
        
        return [global_max_length, global_min_count]