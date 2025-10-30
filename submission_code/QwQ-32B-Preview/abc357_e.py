def main():
    import sys
    sys.setrecursionlimit(2 * 10**5 + 100)
    
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    a = list(map(int, N_and_rest[1:N+1]))
    
    visited = [False] * N
    cycle_ID = [-1] * N
    cycle_T = {}
    cycle_size = {}
    cycle_id = 0
    
    for u in range(N):
        if not visited[u]:
            path = []
            current = u
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = a[current] - 1  # Convert to 0-indexed
            if current in path:
                cycle_start_index = path.index(current)
                cycle_vertices = path[cycle_start_index:]
                C = len(cycle_vertices)
                cycle_size[cycle_id] = C
                cycle_T[cycle_id] = len(path) - C
                for v in cycle_vertices:
                    cycle_ID[v] = cycle_id
                cycle_id += 1
            else:
                existing_cycle_id = cycle_ID[current]
                cycle_T[existing_cycle_id] += len(path)
    
    total_pairs = 0
    for cid in cycle_size:
        C = cycle_size[cid]
        T = cycle_T[cid]
        total_pairs += T * (C + 1) + C * C
    
    print(total_pairs)

if __name__ == "__main__":
    main()