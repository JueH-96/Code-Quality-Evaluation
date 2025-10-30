import sys
import math

def find_cycles(P, N):
    visited = [False] * (N + 1)
    cycles = []
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            cycles.append(cycle)
    return cycles

def lcm(a, b):
    return a * b // math.gcd(a, b)

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = int(data[2 + i - 1])
    
    cycles = find_cycles(P, N)
    
    # Compute LCM of cycle lengths
    L = 1
    for cycle in cycles:
        cycle_len = len(cycle)
        L = lcm(L, cycle_len)
    
    # Compute M = 2^K mod L
    M = pow(2, K, L)
    
    # Create a new permutation after M operations
    new_P = [0] * (N + 1)
    for cycle in cycles:
        cycle_len = len(cycle)
        step = M % cycle_len
        for idx, x in enumerate(cycle):
            new_idx = (idx + step) % cycle_len
            new_P[x] = cycle[new_idx]
    
    # Print the result
    print(' '.join(map(str, new_P[1:N+1])))

if __name__ == "__main__":
    main()