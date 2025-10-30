from typing import List
import sys
sys.setrecursionlimit(300000)

class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        # Build an undirected tree.
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # We use a DFS DP approach.
        # For every node u, we compute a pair (dp0, dp1) where:
        #   dp0 = maximum weight we can get from the subtree rooted at u,
        #         under the assumption that the edge from u's parent to u is NOT taken.
        #   dp1 = maximum weight we can get from the subtree rooted at u,
        #         under the assumption that the edge from u's parent to u IS taken.
        #
        # If the edge from u's parent is taken then that counts as one incident edge at u.
        # Consequently, u can choose at most (k-1) additional edges to its children.
        # If the parent's edge is not taken then u can choose at most k edges toward children.
        #
        # When processing a node u we consider every child v (v != parent)
        # with the connecting edge weight w. We have two options for the edge (u,v):
        #   • NOT use it. Then the child's subtree contributes dp0(v).
        #   • Use it. Then the parent's edge (u,v) adds weight w, and the child's node v 
        #     “sees” that its parent edge is selected, so its subtree’s optimum is dp1(v).
        # The extra gain if we choose the edge (u,v) is:
        #      delta = (dp1(v) + w) - dp0(v).
        #
        # Since u has a “budget” (capacity) on how many children edges it can choose – 
        # capacity = k (if parent's edge not taken) or k-1 (if parent's edge is taken) –
        # the optimal decision is to take, among its children, those edges with the highest positive
        # delta gains, but not more than its capacity.
        #
        # Then the DP for u is given by:
        #    dp(u, state) = (sum over all children of dp0(v)) + (sum of the best positive deltas,
        #                    taking at most capacity, where capacity = k if state==0 and k-1 if state==1)
        
        def dfs(u: int, parent: int) -> (int, int):
            base_sum = 0   # Sum of dp0 values over all children.
            gains = []     # List to store the extra gain ("delta") if we choose an edge to a child.
            for v, w in graph[u]:
                if v == parent:
                    continue
                child_dp0, child_dp1 = dfs(v, u)
                base_sum += child_dp0
                # If we decide to connect u to v, then we get extra w plus the child's subtree advantage
                # when the edge from u to v is chosen (child_dp1) instead of not choosing it (child_dp0).
                delta = (child_dp1 + w) - child_dp0
                if delta > 0:
                    gains.append(delta)
            # Sort the potential gains in descending order.
            gains.sort(reverse=True)
            
            # For state = 0: u's parent edge is not taken, so capacity to choose among children is k.
            cap0 = k
            sum0 = base_sum + sum(gains[:min(cap0, len(gains))])
            # For state = 1: u's parent's edge is taken, so capacity= k-1.
            cap1 = k - 1
            sum1 = base_sum + (sum(gains[:min(cap1, len(gains))]) if cap1 > 0 else 0)
            
            return sum0, sum1
        
        dp0, _ = dfs(0, -1)
        return dp0