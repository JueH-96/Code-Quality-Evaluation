from collections import deque
import sys
from typing import List

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        # Binary search over edge weights.
        low = min(w for _, _, w in edges)
        high = max(w for _, _, w in edges)
        ans = -1

        def feasible(X: int) -> bool:
            # Build the reversed graph from the induced graph (only edges with weight <= X).
            # In the original graph an edge [u, v, w] means u->v. In the reversed graph we add edge: v -> u.
            R = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= X:
                    R[v].append(u)

            # Run BFS starting at node 0 in the reversed graph.
            # (This exactly computes: from 0, which vertices can be reached? Equivalently in the original graph each vertex can reach 0.)
            d = [-1] * n
            dq = deque()
            d[0] = 0
            dq.append(0)
            while dq:
                cur = dq.popleft()
                for nxt in R[cur]:
                    if d[nxt] == -1:
                        d[nxt] = d[cur] + 1
                        dq.append(nxt)
            # If some node did not get reached then condition (1) fails.
            for i in range(n):
                if d[i] == -1:
                    return False

            # Group vertices by their BFS level.
            levels = {}
            max_level = 0
            for i in range(n):
                lvl = d[i]
                if lvl not in levels:
                    levels[lvl] = []
                levels[lvl].append(i)
                if lvl > max_level:
                    max_level = lvl

            # Precompute for each vertex v (v != 0) the list of candidate parents.
            # (If an edge (u,v) exists in R then in the original graph we had u->v.
            # To “go down” one BFS level, we require d[u] == d[v]-1.)
            cand = [[] for _ in range(n)]
            for v in range(1, n):
                for u in R[v]:
                    if d[u] == d[v] - 1:
                        cand[v].append(u)
                if len(cand[v]) == 0:
                    return False

            # For each level l >= 1, we need to “assign” for each node in level l a parent in level (l-1)
            # while ensuring that no vertex in level (l-1) is used more than threshold times.
            sys.setrecursionlimit(10**6)
            for lvl in range(1, max_level + 1):
                if lvl not in levels:
                    continue
                left_nodes = levels[lvl]  # vertices needing a parent (in level lvl)
                # Only vertices in the previous level (lvl-1) can serve as candidate parents.
                match = { u: [] for u in levels[lvl - 1] }  # for each candidate parent, the list of children assigned so far

                # DFS routine to try to assign (match) a left node v.
                # We use a visited set (for vertices in the right side) to avoid cycles.
                def dfs(v, visited):
                    for u in cand[v]:
                        if u in visited:
                            continue
                        visited.add(u)
                        if len(match[u]) < threshold:
                            match[u].append(v)
                            return True
                        # u is full; try to reassign one of the nodes (already matched to u)
                        for i_idx, w in enumerate(match[u]):
                            if dfs(w, visited):
                                match[u][i_idx] = v
                                return True
                    return False

                # Try to match every vertex v in the current level.
                for v in left_nodes:
                    if not dfs(v, set()):
                        return False
            return True

        # Binary search for the minimum possible X for which a feasible selection exists.
        while low <= high:
            mid = (low + high) // 2
            if feasible(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1

        return ans