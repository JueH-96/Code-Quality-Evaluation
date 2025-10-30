def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    # adjacency matrix starts from data[1]
    A = data[1:]
    
    # Convert gathered input into a 2D adjacency matrix
    adj_matrix = []
    idx = 0
    for i in range(N):
        row = list(map(int, A[idx:idx+N]))
        idx += N
        adj_matrix.append(row)
    
    # For each vertex i, find vertices j where adj_matrix[i][j] == 1
    for i in range(N):
        connected = []
        for j in range(N):
            if adj_matrix[i][j] == 1:
                connected.append(j+1)
        if connected:
            print(" ".join(map(str, connected)))
        else:
            print()

# Do not forget to call main
if __name__ == "__main__":
    main()