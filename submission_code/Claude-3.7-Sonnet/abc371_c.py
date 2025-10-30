import itertools

def main():
    # Parse input
    N = int(input())
    
    # Read graph G
    M_G = int(input())
    G = set()
    for _ in range(M_G):
        u, v = map(int, input().split())
        G.add((min(u, v), max(u, v)))
    
    # Read graph H
    M_H = int(input())
    H = set()
    for _ in range(M_H):
        a, b = map(int, input().split())
        H.add((min(a, b), max(a, b)))
    
    # Parse the costs A_{i,j}
    costs = {}
    for i in range(1, N):
        row = list(map(int, input().split()))
        for j in range(i+1, N+1):
            costs[(i, j)] = row[j-i-1]
    
    min_cost = float('inf')
    
    # Try all permutations
    for P in itertools.permutations(range(1, N+1)):
        cost = 0
        
        # Check each possible edge
        for i in range(1, N+1):
            for j in range(i+1, N+1):
                edge_in_G = (min(i, j), max(i, j)) in G
                p_i, p_j = P[i-1], P[j-1]
                p_min, p_max = min(p_i, p_j), max(p_i, p_j)
                edge_in_H = (p_min, p_max) in H
                
                # If edge status doesn't match, we need to toggle it
                if edge_in_G != edge_in_H:
                    cost += costs[(p_min, p_max)]
        
        min_cost = min(min_cost, cost)
    
    print(min_cost)

if __name__ == "__main__":
    main()