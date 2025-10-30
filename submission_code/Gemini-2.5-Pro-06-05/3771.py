import sys

# Set a higher recursion limit for deep graphs, although unlikely with 26 nodes.
sys.setrecursionlimit(2000)

class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k == 0:
            return True
        
        n = len(s)
        
        first = {}
        last = {}
        for i, char in enumerate(s):
            if char not in first:
                first[char] = i
            last[char] = i
            
        unique_chars = sorted(list(first.keys()))
        if not unique_chars:
            return True

        char_to_idx = {char: i for i, char in enumerate(unique_chars)}
        num_unique_chars = len(unique_chars)
        
        # Build dependency graph: c -> d if d appears in the minimal block of c
        adj = [[] for _ in range(num_unique_chars)]
        
        for i in range(num_unique_chars):
            char_c = unique_chars[i]
            start_c, end_c = first[char_c], last[char_c]
            
            seen_in_block = set()
            for j in range(start_c, end_c + 1):
                char_d = s[j]
                if char_d != char_c and char_d not in seen_in_block:
                    idx_d = char_to_idx[char_d]
                    adj[i].append(idx_d)
                    seen_in_block.add(char_d)

        # Find Strongly Connected Components (SCCs) using Tarjan's algorithm
        ids = [-1] * num_unique_chars
        low = [-1] * num_unique_chars
        on_stack = [False] * num_unique_chars
        stack = []
        at = 0
        scc_list = []
        scc_map = [-1] * num_unique_chars

        def dfs(node):
            nonlocal at
            stack.append(node)
            on_stack[node] = True
            ids[node] = low[node] = at
            at += 1

            for neighbor in adj[node]:
                if ids[neighbor] == -1:
                    dfs(neighbor)
                if on_stack[neighbor]:
                    low[node] = min(low[node], low[neighbor])
            
            if ids[node] == low[node]:
                scc_id = len(scc_list)
                scc = []
                while stack:
                    popped_node = stack.pop()
                    on_stack[popped_node] = False
                    scc_map[popped_node] = scc_id
                    scc.append(popped_node)
                    if popped_node == node: break
                scc_list.append(scc)
        
        for i in range(num_unique_chars):
            if ids[i] == -1:
                dfs(i)
        
        num_sccs = len(scc_list)

        # Identify terminal SCCs (those with no outgoing edges in the condensation graph)
        has_outgoing_edge = [False] * num_sccs
        for u in range(num_unique_chars):
            scc_u_id = scc_map[u]
            if has_outgoing_edge[scc_u_id]:
                continue
            for v in adj[u]:
                scc_v_id = scc_map[v]
                if scc_u_id != scc_v_id:
                    has_outgoing_edge[scc_u_id] = True
                    break
        
        count = 0
        for scc_id in range(num_sccs):
            if not has_outgoing_edge[scc_id]:
                # This is a terminal SCC, representing a special group of characters.
                # Check if its span covers the entire string.
                min_f = n
                max_l = -1
                
                scc = scc_list[scc_id]
                for node_idx in scc:
                    char = unique_chars[node_idx]
                    min_f = min(min_f, first[char])
                    max_l = max(max_l, last[char])
                
                # A special substring cannot be the entire string s.
                if max_l - min_f + 1 < n:
                    count += 1
                    
        return count >= k