#!/usr/bin/env python3
import sys,sys
import sys
def main():
    # Explanation:
    #
    # A "Snowflake Tree" is defined by taking a center vertex c,
    # then choosing a positive integer x (the number of arms) and y (the number of leaves per arm)
    # such that we add x vertices (the arms) adjacent to c,
    # and for each arm vertex a, we attach y leaves (neighbors different from c).
    # In the final structure:
    #   • the center has degree exactly x,
    #   • each arm has degree exactly y+1 (its arm‐connection to c plus y leaves),
    #   • and each leaf is degree 1.
    #
    # In the given tree T (with extra vertices and edges), we want to “delete” some vertices
    # so that the remaining induced subgraph is exactly isomorphic to a snowflake tree.
    # Equivalently, we want to choose a subset S of vertices so that S forms a snowflake tree,
    # and we wish to maximize |S|. Then the answer is the minimum number of deletions: N − |S|.
    #
    # How can we “find” a snowflake tree inside T?
    # Notice that in any snowflake tree, the center and its arms are adjacent.
    # Thus if we choose a vertex c as a candidate to be the center, then its arms must be chosen
    # among its neighbors.
    # For a neighbor u (of c) to be a valid arm, it must eventually supply y leaves.
    # In our intended subgraph, the only neighbors of u that will be kept (besides c)
    # are the leaves. Therefore u must have at least one neighbor besides c,
    # i.e. its degree (in T) must be at least 2, so that u has “capacity”
    # cap(u) = deg(u) − 1 (the maximum possible number of leaves we can “reserve” from u).
    #
    # Now, for a fixed candidate center c, let A be the list of eligible arms:
    # all u in N(c) with cap(u) = deg(u) − 1 ≥ 1.
    # Suppose we choose m arms from A. For the induced snowflake tree,
    # each arm must supply exactly y leaves. In order to be valid,
    # we need that for every chosen arm u, we have cap(u) ≥ y.
    # If we choose arms with the m largest capacities, the maximum y we can choose (to use them all)
    # is y = min{cap(u) among the chosen arms} = the m-th highest capacity.
    #
    # Then the snowflake tree will have:
    #   1 (center) + m (arms) + m*y (leaves) = 1 + m*(1+y) vertices.
    #
    # So for a given candidate center c, if we sort the capacities of eligible arms in descending order,
    # then for m from 1 to |A|, a candidate snowflake subgraph size is:
    #   Size = 1 + m*(1 + A_sorted[m-1]),
    # where A_sorted[m-1] is the m-th highest capacity among arms.
    # We then choose the m that yields the largest size.
    #
    # Finally, we choose the best center c among all vertices.
    #
    # Note: It is possible (and sometimes even optimal) to choose a vertex that is not “central”
    # in the original T. For example, for a star T in which the central vertex in T
    # cannot serve as a snowflake center (its neighbors are leaves because they have degree 1),
    # one may choose one of the leaves as center. (Then its only neighbor is the star's center,
    # which will serve as the arm having high capacity.)
    #
    # Once we compute the maximum possible snowflake tree size (say best),
    # the answer is: number to delete = N - best.
    #
    # Implementation details:
    #   • Read the tree; build an adjacency list and compute degrees.
    #   • For every vertex v (from 1 to N) try using it as candidate center.
    #   • Form the list of eligible arms: for each neighbor u of v, if deg(u)>=2 then candidate capacity = deg(u)-1.
    #   • Sort that list in descending order.
    #   • For m=1,...,len(list): candidate size = 1 + m*(1 + (m-th highest capacity)).
    #   • Update the global maximum.
    #
    # Complexity:
    #   The sum over all vertices of the number of neighbors is O(N), and for each vertex
    #   we do a sort on a list of size at most deg(v). In worst-case (for example a star) this is O(N logN), 
    #   which is acceptable for N up to 300k in Python.
    
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    N = n
    # Build graph (1-indexed)
    graph = [[] for _ in range(n+1)]
    deg = [0]*(n+1)
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1

    best = 0  # best snowflake tree size found
    # Try each vertex as candidate center.
    for v in range(1, n+1):
        arms = []
        # For each neighbor u of v, we can use u as an "arm" if it has at least one neighbor besides v.
        for u in graph[v]:
            cap = deg[u] - 1  # available leaves from u if chosen as an arm
            if cap >= 1:
                arms.append(cap)
        if not arms:
            continue
        # Sort capacities in descending order.
        arms.sort(reverse=True)
        k = len(arms)
        # For each possible m (number of arms chosen, from 1 to k),
        # the maximum uniform y possible is the m-th largest capacity (arms[m-1]).
        for i, cap in enumerate(arms):
            m = i + 1
            # The size of the snowflake induced subgraph is: 1 (center) + m (arms) + m*(y) (leaves)
            # with y = cap
            size = 1 + m * (cap + 1)
            if size > best:
                best = size

    # As a fallback, the smallest valid snowflake tree has size 3.
    if best < 3:
        best = 3
    # We cannot retain more vertices than exist.
    best = min(best, n)
    sys.stdout.write(str(n - best))

if __name__ == '__main__':
    main()