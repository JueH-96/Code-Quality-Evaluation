import sys

def solve():
    N = int(sys.stdin.readline())
    
    adj = [[] for _ in range(N)]
    degree = [0] * N
    
    if N < 3: # Constraint: N >= 3
        print(0)
        return

    for _ in range(N - 1):
        u_in, v_in = map(int, sys.stdin.readline().split())
        # Adjust to 0-indexed
        u = u_in - 1
        v = v_in - 1
        
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # This counter will store the sum of directed paths found.
    # Each undirected path x-y will be found as x -> y and y -> x.
    count_directed_paths = 0

    # Iterate through all nodes to find potential start_nodes 'x'
    for i in range(N):
        # 'i' is a potential 'x' endpoint, which must have degree 2 in the tree T
        if degree[i] == 2:
            # For each neighbor of 'x' (call it v_1):
            for neighbor_of_x in adj[i]:
                # v_1 must be the first internal node on the path and must have degree 3 in T
                if degree[neighbor_of_x] == 3:
                    # Start an iterative DFS from v_1.
                    # 'i' (which is 'x') is the previous node for v_1.
                    
                    # Stack for iterative DFS: stores (current_node, previous_node)
                    # Path explored so far is x ... previous_node - current_node
                    stack_dfs = [(neighbor_of_x, i)]

                    while stack_dfs:
                        curr_path_node, prev_path_node = stack_dfs.pop()
                        
                        # Explore neighbors of current_path_node
                        for next_node_in_path in adj[curr_path_node]:
                            if next_node_in_path == prev_path_node: # Don't go back
                                continue

                            if degree[next_node_in_path] == 2:
                                # Found an endpoint 'y' (which is 'next_node_in_path')
                                # The path is x ... prev_path_node - curr_path_node - next_node_in_path (y)
                                # Structure: x(deg 2) - v1(deg 3) - ... - vk(deg 3) - y(deg 2)
                                # 'i' is x. 'neighbor_of_x' is v1.
                                # 'curr_path_node' is vk. 'next_node_in_path' is y.
                                # This path structure ensures length >= 2, so x and y are not adjacent.
                                count_directed_paths += 1
                            elif degree[next_node_in_path] == 3:
                                # Continue the chain of degree-3 nodes
                                stack_dfs.append((next_node_in_path, curr_path_node))
                            # Else (degree[next_node_in_path] is not 2 or 3):
                            #   The chain is broken. Stop exploring this branch.
                            #   (Implicitly done by not adding to stack_dfs)
                    
    # Each valid path (x, y) is found twice: once when starting from x,
    # and once when starting from y. So, divide by 2.
    print(count_directed_paths // 2)

solve()