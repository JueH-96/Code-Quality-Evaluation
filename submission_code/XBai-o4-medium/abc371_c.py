import itertools
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    
    M_G = int(input[ptr])
    ptr += 1
    
    g_edges = set()
    for _ in range(M_G):
        u = int(input[ptr])
        v = int(input[ptr + 1])
        ptr += 2
        g_edges.add((u, v))
    
    M_H = int(input[ptr])
    ptr += 1
    
    h_edges = set()
    for _ in range(M_H):
        a = int(input[ptr])
        b = int(input[ptr + 1])
        ptr += 2
        h_edges.add((a, b))
    
    # Initialize cost matrix A
    A = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N):
        elements = list(map(int, input[ptr:ptr + (N - i)]))
        ptr += (N - i)
        idx = 0
        for j in range(i+1, N+1):
            A[i][j] = elements[idx]
            idx += 1
    
    min_cost = float('inf')
    
    for p in itertools.permutations(range(1, N+1)):
        S = set()
        for (u, v) in g_edges:
            x = p[u-1]
            y = p[v-1]
            if x > y:
                x, y = y, x
            S.add((x, y))
        
        current_cost = 0
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                in_H = (i, j) in h_edges
                in_S = (i, j) in S
                if in_H != in_S:
                    current_cost += A[i][j]
        
        if current_cost < min_cost:
            min_cost = current_cost
    
    print(min_cost)

if __name__ == "__main__":
    main()