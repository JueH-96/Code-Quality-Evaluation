def solve():
    import sys
    
    # Read the two points
    p, q = sys.stdin.readline().split()
    
    # Mapping of points to indices
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Distances between adjacent points
    distances = [3, 1, 4, 1, 5, 9]  # A->B, B->C, C->D, D->E, E->F, F->G
    
    # Get the indices of p and q
    idx_p = points.index(p)
    idx_q = points.index(q)
    
    # Ensure idx_p < idx_q for easier summation
    if idx_p > idx_q:
        idx_p, idx_q = idx_q, idx_p
    
    # Sum the distances between p and q
    result = sum(distances[idx_p:idx_q])
    
    # Print the result
    print(result)

# Call the solve() function
if __name__ == "__main__":
    solve()