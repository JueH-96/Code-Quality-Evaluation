import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    n = int(stdin.readline())
    p = list(map(int, stdin.readline().split()))
    a = list(map(int, stdin.readline().split()))
    visited = [False] * (n + 1)  # 1-based

    # Decompose permutation into cycles
    cycles = []
    for i in range(1, n+1):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = p[j-1]  # p is 1-based
            cycles.append(cycle)
    
    for cycle in cycles:
        # Collect elements in the cycle's order
        elements = [a[pos-1] for pos in cycle]
        L = len(cycle)
        sorted_positions = sorted(cycle)
        # Get indices in the cycle for each sorted position
        indices = [cycle.index(pos) for pos in sorted_positions]
        
        best_k = 0
        best_seq = None
        for k in range(L):
            current_seq = [elements[(i + k) % L] for i in indices]
            if best_seq is None or current_seq < best_seq:
                best_k = k
                best_seq = current_seq
        
        # Rotate elements by best_k and assign back to cycle's positions
        rotated = elements[best_k:] + elements[:best_k]
        for i in range(L):
            pos = cycle[i]
            a[pos-1] = rotated[i]
    
    print(' '.join(map(str, a)))

if __name__ == '__main__':
    main()