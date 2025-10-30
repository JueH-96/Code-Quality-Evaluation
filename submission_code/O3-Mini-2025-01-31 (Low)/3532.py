class Solution:
    def timeTaken(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        # Build the undirected tree
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        # weight of a vertex: 1 if odd, 2 if even.
        weight = [1 if (i % 2) == 1 else 2 for i in range(n)]
        
        # We'll compute, for every vertex u:
        # down[u] = maximum marking time in the subtree under u, 
        #          using the rule: moving from u to neighbor v costs weight[v].
        down = [0] * n
        up = [0] * n
        parent = [-1] * n
        
        # DFS1 (postorder) to compute down[].
        # We choose an arbitrary root (0) for the DFS.
        stack = [0]
        order = []
        while stack:
            u = stack.pop()
            order.append(u)
            for v in g[u]:
                if v == parent[u]:
                    continue
                parent[v] = u
                stack.append(v)
        for u in reversed(order):
            best = 0
            for v in g[u]:
                if v == parent[u]:
                    continue
                candidate = weight[v] + down[v]
                if candidate > best:
                    best = candidate
            down[u] = best
        
        # Build children list from the parent pointers.
        children = [[] for _ in range(n)]
        for u in range(n):
            for v in g[u]:
                if v == parent[u]:
                    continue
                children[u].append(v)
        
        # DFS2 (preorder) to compute up[].
        # For each node u, we wish to compute what is the best distance
        # if we go upward (through u's parent, and possibly use a sibling branch).
        # To do that efficiently, for each node u we first determine the top two values
        # from children[u] of (weight[child] + down[child]). Then for a child,
        # if it came from the best candidate, use the second best for sibling path.
        order = [0]
        up[0] = 0  # root has no upward value.
        while order:
            u = order.pop()
            # determine two best candidates among u's children.
            best1 = best2 = -1
            best1_child = -1
            for v in children[u]:
                cand = weight[v] + down[v]
                if cand > best1:
                    best2 = best1
                    best1 = cand
                    best1_child = v
                elif cand > best2:
                    best2 = cand
            # For each child v of u, compute its up value.
            for v in children[u]:
                # Option 1: Go upward from u.
                candidate1 = up[u] + weight[u]
                # Option 2: If there is a sibling branch:
                # If v achieved best1, then the best candidate from u’s children (other than v) is best2.
                candidate2 = 0
                if v == best1_child:
                    if best2 != -1:
                        candidate2 = best2 + weight[u]
                else:
                    if best1 != -1:
                        candidate2 = best1 + weight[u]
                up[v] = max(candidate1, candidate2)
                order.append(v)
        
        # For each vertex u, the time taken when starting with u marked at t = 0
        # is the maximum distance reachable from u: 
        # note that in a path originating at u we do not add weight[u],
        # but our DP computed exactly that.
        res = [max(down[i], up[i]) for i in range(n)]
        return res

# The following is a standalone solution that reads from input and writes output.
def solve():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    data = list(map(int, data))
    # Expected input format:
    # First integer: n (number of nodes)
    # Next n-1 pairs: each two integers u and v describing an edge.
    n = data[0]
    edges = []
    idx = 1
    for _ in range(n - 1):
        u = data[idx]
        v = data[idx + 1]
        idx += 2
        edges.append([u, v])
    sol = Solution()
    ans = sol.timeTaken(edges)
    sys.stdout.write(" ".join(map(str, ans)))

"""
Explanation:

For each starting node s, the marking time for any other node v is given by
  d(s, v) = sum( weight[w] for w in the unique s→v path, excluding s).
In our DP we compute for every vertex u the maximum such distance (over v)
which is the time when the last node gets marked, i.e. the eccentricity of u.

The two‐DFS approach (“down” and “up”) allows us to compute all eccentricities
in O(n) time.
  
Examples:
 Input: edges = [[0,1],[0,2]]
 Output: [2,4,3]

 Input: edges = [[0,1]]
 Output: [1,2]
  
This solution meets the problem specifications.
"""