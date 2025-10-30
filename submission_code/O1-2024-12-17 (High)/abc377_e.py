def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    P_list = list(map(int, data[2:] ))
    
    # Convert to 0-based for convenience
    p = [x - 1 for x in P_list]
    
    visited = [False] * N
    answer = [0] * N
    
    # For each unvisited node, find its cycle
    for start in range(N):
        if not visited[start]:
            cycle = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = p[cur]
            
            L = len(cycle)
            if L == 1:
                # A cycle of length 1 never changes under any power
                answer[cycle[0]] = cycle[0] + 1
            else:
                # Compute 2^K mod L
                steps = pow(2, K, L)
                # Assign final positions for all elements in this cycle
                for i in range(L):
                    answer[cycle[i]] = cycle[(i + steps) % L] + 1
    
    print(" ".join(map(str, answer)))

# Do not forget to call main()
main()