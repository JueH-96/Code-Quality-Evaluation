# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    # Read the number of used product names
    N = int(sys.stdin.readline())

    # If N is 0, no product names are used, so the NG list can be empty.
    # The problem constraints state N >= 1, but handling N=0 is good practice.
    if N == 0:
        print(0)
        return

    # Read the N product names and store them in a set for efficient lookup
    U = set(sys.stdin.readline().strip() for _ in range(N))

    # Build adjacency list for the directed graph where edges are product names (u -> v if "uv" in U)
    adj = defaultdict(list)
    # Build adjacency list for the undirected graph to find Weakly Connected Components (WCC)
    undirected_adj = defaultdict(list)
    # Keep track of all characters (nodes) that appear in the input strings (have edges incident to them)
    nodes_with_edges_set = set()

    # Populate the graph data structures
    for s in U:
        u, v = s[0], s[1]
        adj[u].append(v)
        undirected_adj[u].append(v)
        undirected_adj[v].append(u) # Add edge in both directions for undirected graph
        nodes_with_edges_set.add(u)
        nodes_with_edges_set.add(v)

    # Set to keep track of nodes visited during WCC traversal
    visited_wcc = set()
    # Variable to accumulate the total minimum number of strings in the NG list
    total_paths = 0

    # Iterate through all nodes that are involved in at least one edge
    # Convert the set to a list to iterate while modifying visited_wcc
    for start_node in list(nodes_with_edges_set):
        # If this node hasn't been visited yet, it's the start of a new WCC containing edges
        if start_node not in visited_wcc:
            # Find all nodes belonging to this WCC using BFS on the undirected graph
            component_nodes = set()
            q = deque([start_node])
            visited_wcc.add(start_node)
            component_nodes.add(start_node)

            while q:
                u = q.popleft()
                # Traverse neighbors in the undirected graph
                for v in undirected_adj[u]:
                    if v not in visited_wcc:
                        visited_wcc.add(v)
                        component_nodes.add(v)
                        q.append(v)

            # Found a component (set of nodes). Now identify the original directed edges *within* this component from the set U.
            # We need these edges to calculate degrees within the component.
            component_edges = set()
            out_degree_comp = defaultdict(int)
            in_degree_comp = defaultdict(int)

            # Iterate through all nodes found in this component
            for u in component_nodes:
                 # Check original directed edges starting from u (only if u has outgoing edges in the original graph)
                 if u in adj:
                     for v in adj[u]:
                         if v in component_nodes: # If the destination node v is also in this component
                             # This edge (u,v) from U is entirely within this component
                             # Use tuple (u, v) to represent the edge directionally
                             component_edges.add((u, v))
                             # Calculate degrees based only on edges within this component
                             out_degree_comp[u] += 1
                             in_degree_comp[v] += 1

            # Ensure the component actually contains edges from U.
            # This check should pass if we start from a node in nodes_with_edges_set.
            # A component found by WCC traversal might contain nodes reachable from nodes
            # with edges, but which themselves have no edges within U. But if we start
            # from a node *with* an edge, the component will contain at least that edge.
            if not component_edges:
                 continue

            # Determine if the component is balanced within itself.
            # A component is balanced if for every node v in the component, its indegree equals its outdegree,
            # considering only edges within that component. This means D_comp = 0.
            is_balanced_comp = True
            # Check balance for all nodes that are part of this component.
            # defaultdict(int) gives 0 for nodes with no edges incident in the component, which is correct.
            for node in component_nodes:
                 if out_degree_comp[node] != in_degree_comp[node]:
                      is_balanced_comp = False
                      break

            if is_balanced_comp:
                # If the component is balanced and has edges, it can be covered by 1 path (an Eulerian circuit).
                total_paths += 1
            else:
                # The component is unbalanced.
                # The minimum number of paths required for an unbalanced component
                # is the maximum of the number of 'sources' (nodes with in-degree 0, out-degree > 0)
                # and 'sinks' (nodes with out-degree 0, in-degree > 0) within the component.
                # These represent points where paths must start or end respectively.
                # Note: A node V in the component might have in_degree_comp(V) == 0 and out_degree_comp(V) == 0.
                # These nodes don't start or end paths within this component's coverage requirements
                # unless they are isolated nodes with self-loops which is covered by in>0 and out>0 case.
                # We only count nodes that are "pure" sources or sinks *within this component*.
                n_in0_comp = 0 # Count nodes V in component with in_degree_comp(V) == 0 and out_degree_comp(V) > 0
                n_out0_comp = 0 # Count nodes V in component with out_degree_comp(V) == 0 and in_degree_comp(V) > 0

                for node in component_nodes:
                    if in_degree_comp[node] == 0 and out_degree_comp[node] > 0:
                         n_in0_comp += 1
                    if out_degree_comp[node] == 0 and in_degree_comp[node] > 0:
                         n_out0_comp += 1
                
                # Add the maximum count of 'sources' or 'sinks' to the total paths.
                # If a component is unbalanced and has edges, max(n_in0_comp, n_out0_comp) is guaranteed to be >= 1.
                total_paths += max(n_in0_comp, n_out0_comp)

    # Print the final minimum number of strings
    print(total_paths)

solve()