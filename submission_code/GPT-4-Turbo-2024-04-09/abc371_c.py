import sys
import itertools

def read_int():
    return int(sys.stdin.readline().strip())

def read_ints():
    return map(int, sys.stdin.readline().strip().split())

def main():
    N = read_int()
    M_G = read_int()
    edges_G = set()
    for _ in range(M_G):
        u, v = read_ints()
        if u > v:
            u, v = v, u
        edges_G.add((u, v))
    
    M_H = read_int()
    edges_H = set()
    for _ in range(M_H):
        a, b = read_ints()
        if a > b:
            a, b = b, a
        edges_H.add((a, b))
    
    cost_matrix = {}
    for i in range(1, N):
        costs = list(read_ints())
        for j in range(i + 1, N + 1):
            cost_matrix[(i, j)] = costs[j - i - 1]
    
    min_cost = float('inf')
    
    # Try all permutations of vertices in H to match G
    for perm in itertools.permutations(range(1, N + 1)):
        current_cost = 0
        permuted_edges_H = set()
        
        # Create the permuted edge set of H
        for a, b in edges_H:
            pa, pb = perm[a - 1], perm[b - 1]
            if pa > pb:
                pa, pb = pb, pa
            permuted_edges_H.add((pa, pb))
        
        # Calculate the cost to make permuted H match G
        for i in range(1, N):
            for j in range(i + 1, N + 1):
                if ((i, j) in edges_G) != ((i, j) in permuted_edges_H):
                    current_cost += cost_matrix[(i, j)]
        
        # Update the minimum cost found
        min_cost = min(min_cost, current_cost)
    
    print(min_cost)

if __name__ == "__main__":
    main()