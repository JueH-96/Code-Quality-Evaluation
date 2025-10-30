def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    # Initialize row, column, and diagonal counts
    row_counts = [0] * N
    col_counts = [0] * N
    diag1_count = 0
    diag2_count = 0
    
    # Create a dictionary to map A_i to its cell position
    cell_map = {}
    for i in range(N):
        for j in range(N):
            cell_map[N * i + j + 1] = (i, j)
    
    for turn in range(T):
        a = A[turn]
        if a not in cell_map:
            continue
        i, j = cell_map[a]
        row_counts[i] += 1
        if row_counts[i] == N:
            print(turn + 1)
            return
        col_counts[j] += 1
        if col_counts[j] == N:
            print(turn + 1)
            return
        if i == j:
            diag1_count += 1
            if diag1_count == N:
                print(turn + 1)
                return
        if i + j == N - 1:
            diag2_count += 1
            if diag2_count == N:
                print(turn + 1)
                return
    print(-1)

if __name__ == "__main__":
    main()