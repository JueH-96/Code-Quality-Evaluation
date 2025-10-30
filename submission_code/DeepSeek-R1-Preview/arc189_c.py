def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1]) - 1  # Convert to 0-based index
    
    A = list(map(int, data[2:N+2]))
    B = list(map(int, data[N+2:2*N+2]))
    
    P = list(map(lambda x: int(x)-1, data[2*N+2:3*N+2]))
    Q = list(map(lambda x: int(x)-1, data[3*N+2:4*N+2]))
    
    # Function to find the cycle containing X for a given permutation
    def find_cycle(perm, X):
        visited = set()
        cycle = []
        current = X
        while True:
            if current in visited:
                break
            visited.add(current)
            cycle.append(current)
            current = perm[current]
        # Check if the cycle includes X
        if X in cycle:
            # Rebuild the cycle starting from X
            cycle = []
            current = X
            while True:
                cycle.append(current)
                current = perm[current]
                if current == X:
                    break
            return cycle
        else:
            return []
    
    cycle_p = find_cycle(P, X)
    cycle_q = find_cycle(Q, X)
    
    # Check if all required boxes can reach X
    impossible = False
    for i in range(N):
        if A[i] > 0 and i not in cycle_p:
            impossible = True
        if B[i] > 0 and i not in cycle_q:
            impossible = True
    if impossible:
        print(-1)
        return
    
    # If X is the only box and no balls, output 0
    if all(a == 0 for a in A) and all(b == 0 for b in B):
        print(0)
        return
    
    # Create position dictionaries
    pos_p = {node: idx for idx, node in enumerate(cycle_p)}
    pos_q = {node: idx for idx, node in enumerate(cycle_q)}
    
    S = set()
    
    for i in range(N):
        if A[i] > 0:
            idx = pos_p[i]
            path = cycle_p[idx:]
            for node in path:
                if node != X:
                    S.add(node)
        if B[i] > 0:
            idx = pos_q[i]
            path = cycle_q[idx:]
            for node in path:
                if node != X:
                    S.add(node)
    
    print(len(S))

if __name__ == "__main__":
    main()