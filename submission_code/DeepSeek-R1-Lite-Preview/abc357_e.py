def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    a = list(map(int, N_and_rest[1:N+1]))
    
    # Convert to 0-based indexing
    a = [x-1 for x in a]
    
    group_id = [ -1 ] * N
    is_in_cycle = [ False ] * N
    visited = [ False ] * N
    current_group = 0
    
    for u in range(N):
        if not visited[u]:
            path = []
            v = u
            while True:
                if visited[v]:
                    if v in path:
                        # Found a cycle
                        cycle_start = path.index(v)
                        cycle_size = len(path) - cycle_start
                        # Assign group ID and mark as in cycle
                        for node in path[cycle_start:]:
                            group_id[node] = current_group
                            is_in_cycle[node] = True
                        current_group += 1
                    break
                else:
                    visited[v] = True
                    path.append(v)
                    v = a[v]
            # Assign group ID to the remaining nodes in the path that are not in the cycle
            for node in path:
                if group_id[node] == -1:
                    group_id[node] = current_group - 1
    
    # Now, for each group, find m (cycle size) and k (number of tree vertices)
    from collections import defaultdict
    group_info = defaultdict(lambda: {'m':0, 'k':0})
    for u in range(N):
        g = group_id[u]
        if is_in_cycle[u]:
            group_info[g]['m'] += 1
        else:
            group_info[g]['k'] += 1
    
    total_pairs = 0
    for g in group_info:
        m = group_info[g]['m']
        k = group_info[g]['k']
        total_pairs += m * m + k * (m + 1)
    
    print(total_pairs)

if __name__ == '__main__':
    main()