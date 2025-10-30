def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Build the graph using an adjacency list.
    # For each info "A B X Y", we record that B = A + (X, Y)
    # and we also record the reverse edge "A = B + (-X, -Y)".
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        a = int(next(it))
        b = int(next(it))
        dx = int(next(it))
        dy = int(next(it))
        graph[a].append((b, dx, dy))
        graph[b].append((a, -dx, -dy))
    
    # We need to assign positions to people.
    # Only the connected component that contains person 1 is fixed since person 1 is given at (0, 0).
    # For any component that does not have a fixed person's coordinate, the positions cannot be uniquely determined.
    
    # Initialize positions: None means not uniquely determined (or not yet assigned).
    positions = [None] * (n+1)
    # Person 1 is fixed at (0,0)
    positions[1] = (0, 0)
    
    # We also need to traverse all components to (i) propagate coordinates where possible and (ii)
    # mark a component as "undecidable" if none of its nodes got a fixed position.
    visited = [False] * (n+1)
    
    # Iterate over all nodes. For each unvisited node, perform a DFS/BFS.
    # If in that component at least one node gets a coordinate (i.e., it's connected to a fixed person),
    # then we can propagate absolute positions for all nodes in that component.
    # Otherwise, the positions in that component remain "undecidable".
    for i in range(1, n+1):
        if not visited[i]:
            # Track the nodes in the current component.
            component = []
            # Flag to check if any node in this component is fixed.
            fixed_component = (positions[i] is not None)
            # We'll use a stack for DFS.
            stack = [i]
            
            while stack:
                cur = stack.pop()
                if visited[cur]:
                    continue
                visited[cur] = True
                component.append(cur)
                for nb, dx, dy in graph[cur]:
                    # If the current node has a fixed position, then the neighbor's position can be computed.
                    if positions[cur] is not None:
                        new_pos = (positions[cur][0] + dx, positions[cur][1] + dy)
                        if positions[nb] is None:
                            positions[nb] = new_pos
                        # (The problem guarantees consistency so we need not check if a previously assigned value is different.)
                    # If the neighbor already had a fixed position, mark that this component is fixed.
                    if positions[nb] is not None:
                        fixed_component = True
                    # Continue traversing.
                    if not visited[nb]:
                        stack.append(nb)
            
            # If this component does not have a fixed coordinate, then the positions remain "undecidable".
            # We mark them explicitly as None.
            if not fixed_component:
                for node in component:
                    positions[node] = None
    
    # Output results.
    out_lines = []
    for i in range(1, n+1):
        if positions[i] is None:
            out_lines.append("undecidable")
        else:
            out_lines.append(f"{positions[i][0]} {positions[i][1]}")
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()