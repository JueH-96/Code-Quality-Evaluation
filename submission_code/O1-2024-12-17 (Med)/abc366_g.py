def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    edges = input_data[2:]

    # Build adjacency list
    adjacency = [[] for _ in range(N)]
    idx = 0
    for _ in range(M):
        u = int(edges[idx]) - 1
        v = int(edges[idx+1]) - 1
        idx += 2
        adjacency[u].append(v)
        adjacency[v].append(u)

    # Quick check: if any vertex has degree 1, answer is No immediately
    degrees = [len(adjacency[i]) for i in range(N)]
    for d in degrees:
        if d == 1:
            print("No")
            return

    # If there are no edges at all, we can assign 1 to all vertices
    if M == 0:
        print("Yes")
        print(" ".join(["1"]*N))
        return

    # We will assign an integer X[v] to each vertex v (None if not decided yet).
    X = [None]*N

    # We need to process connected components that actually have edges.
    visited = [False]*N

    # Gather connected components via DFS/BFS
    from collections import deque

    def bfs_component(start):
        """ Return all vertices in the connected component of 'start'. """
        queue = deque([start])
        comp = []
        visited[start] = True
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        return comp

    def solve_component(vertices):
        """
        We have the subgraph induced by 'vertices'.
        We want, for each vertex v in 'vertices' with deg>=1, the XOR of X[u] for u in adjacency[v] = 0.
        We'll do an iterative assignment:
        - If a vertex v already has all neighbors assigned, check their XOR is 0 (contradiction if not).
        - If exactly one neighbor is unassigned, we can deduce that one neighbor.
        - If more than one neighbor is unassigned, we skip for now.
        - If still unassigned remain in a cycle, pick one unassigned vertex, set X=1, propagate again.
        If at any point we must assign X=0, we declare contradiction => return False.
        Otherwise return True on success.
        """

        # A set for quick membership
        vert_set = set(vertices)

        # Precompute adjacency restricted to this component (since some components might be disconnected).
        # Also skip degree-0 in this component if it has no edges inside this component.
        # But typically we'd only worry about vertices that actually have edges in the subgraph.
        sub_adj = {}
        for v in vertices:
            # neighbors that are also in 'vertices'
            nb = [w for w in adjacency[v] if w in vert_set]
            sub_adj[v] = nb

        # We do iterative propagation
        changed_global = True
        assigned_count = sum(1 for v in vertices if X[v] is not None)

        # Function to attempt propagation until no immediate changes
        def propagate():
            changed = True
            while changed:
                changed = False
                for v in vertices:
                    nb = sub_adj[v]
                    # If v has no neighbors in the component, skip (no constraint here)
                    if len(nb) == 0:
                        continue
                    # Check how many neighbors are assigned
                    assigned = [u for u in nb if X[u] is not None]
                    unassigned = [u for u in nb if X[u] is None]
                    # XOR of assigned neighbors
                    partial = 0
                    for u in assigned:
                        partial ^= X[u]
                    if len(unassigned) == 0:
                        # Then partial must be 0, otherwise contradiction
                        if partial != 0:
                            return False
                    elif len(unassigned) == 1:
                        # We can deduce the label for that single unassigned neighbor
                        w = unassigned[0]
                        # We need partial ^ X[w] = 0 => X[w] = partial
                        if partial == 0:
                            # That would force X[w] = 0 => contradiction
                            return False
                        # If X[w] is not None but different, contradiction
                        if X[w] is not None and X[w] != partial:
                            return False
                        if X[w] is None:
                            X[w] = partial
                            changed = True
                    # else if >=2 unassigned, we can't deduce here yet

            return True

        # Keep trying to assign until all assigned or contradiction
        # Each time we do a full pass of propagate().  If not all assigned,
        # we pick one unassigned (with edges) and set it to 1, then propagate again.
        # We do this up to len(vertices) times at most.
        for _ in range(len(vertices)):
            ok = propagate()
            if not ok:
                return False

            # Check if all neighbors assigned in the subgraph
            unassigned_vertices = [v for v in vertices if X[v] is None and len(sub_adj[v]) > 0]
            if len(unassigned_vertices) == 0:
                # All edges' endpoints are assigned.  We are done in this component
                return True

            # Otherwise pick one unassigned vertex that actually has edges in this component, set X=1
            pick = unassigned_vertices[0]
            X[pick] = 1
            # and loop around again

        # If we exit the loop with unassigned still, we try one last propagate
        ok = propagate()
        if not ok:
            return False

        # If still unassigned remain for vertices that do have edges, it's suspicious,
        # but we might have repeated the process len(vertices) times, so let's confirm:
        unassigned_vertices = [v for v in vertices if X[v] is None and len(sub_adj[v]) > 0]
        if len(unassigned_vertices) > 0:
            return False

        return True

    # Process each connected component
    for i in range(N):
        if not visited[i]:
            comp = bfs_component(i)
            # Check if this component has at least one edge
            # i.e. if any vertex in comp has adjacency inside comp
            has_edge = any(degrees[v] > 0 for v in comp)
            if has_edge:
                # Solve constraints within this component
                if not solve_component(comp):
                    print("No")
                    return

    # At this point, we have assigned consistent labels in all components with edges or found contradiction.
    # Some vertices may still be None if they had degree=0 => just assign 1 to them
    for i in range(N):
        if X[i] is None:
            X[i] = 1  # any non-zero in [1..2^60 - 1]

    # Final check: ensure no label is 0
    for i in range(N):
        if X[i] == 0:
            print("No")
            return

    print("Yes")
    print(" ".join(map(str, X)))


# Do not forget to call main()!
if __name__ == "__main__":
    main()