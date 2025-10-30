import sys

def solve():
    """
    This function solves the problem using a greedy algorithm.
    It iteratively builds the optimal subtree for K=1, 2, ..., N.
    """
    # Increase recursion limit for deep trees, although this implementation is iterative.
    sys.setrecursionlimit(2 * 10**5 + 5)
    
    try:
        input = sys.stdin.readline
        N_str = input()
        if not N_str: return
        N = int(N_str)
    except (IOError, ValueError):
        return

    if N == 1:
        print(0)
        return

    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v, l = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append((v, l))
        adj[v].append((u, l))

    def bfs(start_nodes):
        """
        Performs a multi-source BFS to compute shortest distances from a set of start nodes.
        """
        dists = [-1] * N
        q = []
        for s in start_nodes:
            if dists[s] == -1:
                dists[s] = 0
                q.append(s)
        
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            d = dists[u]
            for v, l in adj[u]:
                if dists[v] == -1:
                    dists[v] = d + l
                    q.append(v)
        return dists

    # T_sub_nodes is the set of vertices in the current optimal subtree
    T_sub_nodes = {0}
    dist_to_T = bfs(T_sub_nodes)
    
    total_len = 0
    
    for k in range(1, N + 1):
        farthest_v = -1
        max_dist = -1
        
        # Find the vertex globally farthest from the current subtree
        for v_candidate in range(N):
            if dist_to_T[v_candidate] > max_dist:
                max_dist = dist_to_T[v_candidate]
                farthest_v = v_candidate

        # If all vertices are in the subtree, distances are 0. The score stabilizes.
        if max_dist == 0:
            print(2 * total_len)
            continue
        
        total_len += max_dist
        print(2 * total_len)

        if k == N:
            break

        # Find the path from the farthest vertex to the current subtree to expand it.
        # This small BFS finds the path and the attachment point.
        q = [(farthest_v, [farthest_v])]
        head = 0
        new_path_nodes = []
        visited_path_bfs = {farthest_v}

        while head < len(q):
            curr, path = q[head]
            head += 1

            if curr in T_sub_nodes:
                new_path_nodes = path
                break
            
            for neighbor, weight in adj[curr]:
                if neighbor not in visited_path_bfs:
                    visited_path_bfs.add(neighbor)
                    new_path_list = list(path)
                    new_path_list.append(neighbor)
                    q.append((neighbor, new_path_list))
        
        # Add all vertices on the new path to our subtree's node set.
        for node in new_path_nodes:
            T_sub_nodes.add(node)
        
        # Update distances to the newly expanded subtree. This is the bottleneck in a naive implementation.
        dist_to_T = bfs(T_sub_nodes)

solve()