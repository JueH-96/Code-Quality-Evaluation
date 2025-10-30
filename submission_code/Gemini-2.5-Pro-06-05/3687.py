import collections
import sys

# It's good practice to increase recursion limit for deep trees.
sys.setrecursionlimit(5 * 10**4 + 5)

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        
        # 1. Build directed adjacency list (tree rooted at 0)
        undir_adj = collections.defaultdict(list)
        for u, v, w in edges:
            undir_adj[u].append((v, w))
            undir_adj[v].append((u, w))

        adj = [[] for _ in range(n)]
        q = collections.deque([0])
        visited = {0}
        while q:
            u = q.popleft()
            for v, w in undir_adj[u]:
                if v not in visited:
                    visited.add(v)
                    adj[u].append((v, w))
                    q.append(v)
        
        # 2. State for the answer. A single node is a path of length 0 with 1 node.
        max_len = 0
        min_nodes = 1
        
        # 3. Helper functions
        def update_global(l, n):
            nonlocal max_len, min_nodes
            if l > max_len:
                max_len = l
                min_nodes = n
            elif l == max_len:
                min_nodes = min(min_nodes, n)

        def is_better(p1, p2):
            # p = (length, nodes)
            # A path is better if it's longer, or same length with fewer nodes.
            return p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] < p2[1])

        # 4. Main DFS logic
        def dfs(u: int) -> dict:
            # paths_from_u: map from end_value to (length, nodes) for special paths starting at u
            paths_from_u = {nums[u]: (0, 1)}
            update_global(0, 1)

            for v, w in adj[u]:
                child_map = dfs(v)
                for val, (l, n) in child_map.items():
                    # A path from a child can be extended if the parent's value is not a duplicate.
                    # The only potential duplicate value is the end_value of the child's path,
                    # because if it were an intermediate value, u would be a descendant of v.
                    if val != nums[u]:
                        new_path = (l + w, n + 1)
                        update_global(new_path[0], new_path[1])
                        
                        # If this new path is better for this end_value, update our map
                        if is_better(new_path, paths_from_u.get(val, (-1, -1))):
                            paths_from_u[val] = new_path
            return paths_from_u
        
        # Start the traversal from the root.
        dfs(0)
        
        return [max_len, min_nodes]