import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    X = [x-1 for x in X]  # Convert to 0-based index
    A = list(map(int, sys.stdin.readline().split()))
    
    visited = [False] * N
    pos_in_cycle = {}
    cycle_length = {}
    
    for i in range(N):
        if not visited[i]:
            current = i
            cycle = []
            while True:
                if visited[current]:
                    break
                visited[current] = True
                cycle.append(current)
                current = X[current]
            L = len(cycle)
            for idx, node in enumerate(cycle):
                pos_in_cycle[node] = idx
                cycle_length[node] = L
    
    result = []
    for i in range(N):
        if cycle_length[i] == 0:
            result.append(A[i])
        else:
            L = cycle_length[i]
            K_mod = K % L
            new_pos = (pos_in_cycle[i] + K_mod) % L
            result.append(A[new_pos])
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()