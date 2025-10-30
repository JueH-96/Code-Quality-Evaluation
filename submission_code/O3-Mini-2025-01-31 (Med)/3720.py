from typing import List
import sys

class Solution:
    def minMaxWeight(self, n: int, edges: List[List[int]], threshold: int) -> int:
        sys.setrecursionlimit(10**6)
        # Explanation:
        # We want to remove some edges so that:
        #  1. Every node has a path to node 0.
        #  2. Each node has at most "threshold" outgoing edges.
        #  3. The maximum edge weight among those remaining is minimized.
        #
        # A spanning arborescence (a directed spanning tree where every node except 0
        # has exactly one incoming edge) ensures condition (1). In the chosen arborescence,
        # a node’s outgoing edges are just its children in the tree. Thus if we can pick,
        # for every node v (except 0), one incoming edge such that the parent (say u)
        # does not provide more than threshold edges (i.e. u is parent to at most threshold nodes),
        # then both connectivity and the outdegree constraint are met.
        #
        # Notice that the chosen edges must have weight <= X for some candidate X and
        # we want to minimize X. So we can binary search on X (taken from the unique weights).
        #
        # For a given X we consider only those edges with weight <= X.
        # Then, for every vertex v (v != 0) we try to assign a parent u (making use
        # of an edge u->v) so that no node u is used more than threshold times.
        #
        # This is exactly a bipartite matching problem where:
        #   - Right side: nodes {1,2,...,n-1} (each requiring one incoming edge).
        #   - Left side: all nodes 0..n-1; each u can supply up to "threshold" edges.
        #   - There's an edge from u (left) to v (right) if there is an edge u->v in the
        #     original graph (with weight <= X).
        #
        # We can check if there is a matching covering every node on the right.
        # To do that we use a DFS-based matching algorithm that handles capacity >1 on left nodes.
        
        # Get sorted unique weights.
        uniqueWeights = sorted({w for _, _, w in edges})
        
        # Given an allowed maximum weight X, check feasibility using a matching algorithm.
        def can(X: int) -> bool:
            # For each right node v (v != 0), build the list of candidate left nodes u such that
            # there is an edge u -> v with weight <= X.
            candidates = [[] for _ in range(n)]
            for u, v, w in edges:
                if w <= X and v != 0:
                    candidates[v].append(u)
            
            # For each left node u, we maintain a list "match[u]" of right nodes v that are currently matched
            # to u. We can use u up to "threshold" times.
            match = [[] for _ in range(n)]
            
            # Standard DFS to try to find an augmenting assignment for a right vertex v.
            # We'll use a "used" array on left nodes (u) per DFS call.
            def try_match(v: int, used: List[bool]) -> bool:
                for u in candidates[v]:
                    if used[u]:
                        continue
                    used[u] = True
                    # If u has capacity spare, assign v to u.
                    if len(match[u]) < threshold:
                        match[u].append(v)
                        return True
                    else:
                        # Otherwise, try to reassign one of the right vertices already matched with u.
                        for i in range(len(match[u])):
                            other = match[u][i]
                            if try_match(other, used):
                                # Found a reassignment; assign the current v to u.
                                match[u][i] = v
                                return True
                return False
            
            count = 0
            # Try to assign every right node (v != 0).
            for v in range(1, n):
                used = [False] * n
                if try_match(v, used):
                    count += 1
                else:
                    return False
            return count == (n - 1)
        
        # Binary search over the sorted unique weights.
        lo, hi = 0, len(uniqueWeights) - 1
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if can(uniqueWeights[mid]):
                ans = uniqueWeights[mid]
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    n1 = 5
    edges1 = [[1,0,1],[2,0,2],[3,0,1],[4,3,1],[2,1,1]]
    threshold1 = 2
    print(sol.minMaxWeight(n1, edges1, threshold1))  # Expected output: 1
    
    # Example 2:
    n2 = 5
    edges2 = [[0,1,1],[0,2,2],[0,3,1],[0,4,1],[1,2,1],[1,4,1]]
    threshold2 = 1
    print(sol.minMaxWeight(n2, edges2, threshold2))  # Expected output: -1
    
    # Example 3:
    n3 = 5
    edges3 = [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[3,4,2],[4,0,1]]
    threshold3 = 1
    print(sol.minMaxWeight(n3, edges3, threshold3))  # Expected output: 2
    
    # Example 4:
    n4 = 5
    edges4 = [[1,2,1],[1,3,3],[1,4,5],[2,3,2],[4,0,1]]
    threshold4 = 1
    print(sol.minMaxWeight(n4, edges4, threshold4))  # Expected output: -1