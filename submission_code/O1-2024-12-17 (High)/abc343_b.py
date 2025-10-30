def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    matrix = []
    idx = 1
    
    # Read the adjacency matrix
    for _ in range(N):
        row = list(map(int, data[idx:idx+N]))
        matrix.append(row)
        idx += N
    
    # For each vertex i, find all vertices j where matrix[i][j] == 1
    for i in range(N):
        neighbors = [str(j + 1) for j in range(N) if matrix[i][j] == 1]
        print(' '.join(neighbors))

# Do not forget to call main() at the end
if __name__ == '__main__':
    main()