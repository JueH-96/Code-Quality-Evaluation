def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = data[1:]  # List of N strings, each of length N
    
    # Prepare an output grid of the same dimensions
    B = [ [None]*N for _ in range(N) ]
    
    # A small helper to rotate (r, c) by 90° clockwise in a 0-based N×N
    # f(r, c) -> (c, N-1-r)
    def rotate_90(r, c):
        return (c, N - 1 - r)
    
    # Apply k times the 90° rotation (k mod 4 handles full cycle)
    def rotate_k_times(r, c, k):
        if k == 0:
            return (r, c)
        elif k == 1:
            return rotate_90(r, c)
        elif k == 2:
            # Rotate twice
            r1, c1 = rotate_90(r, c)
            return rotate_90(r1, c1)
        else:  # k == 3
            # Rotate thrice
            r1, c1 = rotate_90(r, c)
            r2, c2 = rotate_90(r1, c1)
            return rotate_90(r2, c2)
    
    # Fill the final grid
    for r in range(N):
        row_str = M[r]
        for c in range(N):
            # Which "layer" of the matrix is this cell in?
            # (ringIndex is 0-based for the layer, so add 1 for the count)
            ringIndex = min(r, c, N - 1 - r, N - 1 - c)
            rotations = ringIndex + 1  # how many 90° rotations in total
            k = rotations % 4
            nr, nc = rotate_k_times(r, c, k)
            B[nr][nc] = row_str[c]
    
    # Output the result
    print("
".join("".join(row) for row in B))

# Do not forget to call main()
main()