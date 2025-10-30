def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = [0] + list(map(int, data[1:]))
    
    # global visited: 0 = not visited, 1 = processing, 2 = finished.
    state = [0] * (N + 1)
    
    # This will store the cycle if found.
    cycle_result = None

    def dfs(v, path, pos_map):
        nonlocal cycle_result
        state[v] = 1  # Mark as processing.
        path.append(v)
        pos_map[v] = len(path) - 1
        
        nxt = A[v]
        if state[nxt] == 0:
            # Not visited, continue dfs.
            dfs(nxt, path, pos_map)
            if cycle_result is not None:
                return
        elif state[nxt] == 1:
            # Found a cycle.
            start_index = pos_map[nxt]
            cycle_result = path[start_index:]
            return
        # else if state[nxt] == 2, already finished, then no new cycle in this path.
        
        # Pop from current path and mark finished.
        path.pop()
        pos_map.pop(v)
        state[v] = 2
        
    for i in range(1, N+1):
        if state[i] == 0:
            path = []
            pos_map = {}
            dfs(i, path, pos_map)
            if cycle_result is not None:
                break

    if cycle_result is None:
        # According to problem statement, cycle always exists.
        cycle_result = []

    # Print cycle
    # The length is the number of vertices in cycle.
    output = []
    output.append(str(len(cycle_result)))
    output.append(" ".join(map(str, cycle_result)))
    sys.stdout.write("
".join(output))
    
if __name__ == '__main__':
    main()