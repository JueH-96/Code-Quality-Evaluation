# YOUR CODE HERE
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
    diag1_count = 0  # Top-left to bottom-right
    diag2_count = 0  # Top-right to bottom-left
    
    # Create a dictionary to map A_i to its position (i, j)
    pos = {}
    for idx in range(T):
        a = A[idx]
        i = (a - 1) // N + 1
        j = (a - 1) % N + 1
        pos[a] = (i, j)
    
    for turn in range(T):
        a = A[turn]
        i, j = pos[a]
        # Update row count
        row_counts[i-1] += 1
        if row_counts[i-1] == N:
            print(turn + 1)
            return
        # Update column count
        col_counts[j-1] += 1
        if col_counts[j-1] == N:
            print(turn + 1)
            return
        # Update diagonal counts
        if i == j:
            diag1_count += 1
            if diag1_count == N:
                print(turn + 1)
                return
        if i + j == N + 1:
            diag2_count += 1
            if diag2_count == N:
                print(turn + 1)
                return
    print(-1)

if __name__ == "__main__":
    main()