def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    T = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+T]))
    
    row_counts = [0] * (N + 1)
    column_counts = [0] * (N + 1)
    main_diag_count = 0
    anti_diag_count = 0
    
    for turn in range(1, T + 1):
        a = A[turn - 1]
        i = (a - 1) // N + 1
        j = (a - 1) % N + 1
        
        row = i
        column = j
        
        row_counts[row] += 1
        column_counts[column] += 1
        
        if row == column:
            main_diag_count += 1
        if row + column == N + 1:
            anti_diag_count += 1
        
        if (row_counts[row] == N or
            column_counts[column] == N or
            main_diag_count == N or
            anti_diag_count == N):
            print(turn)
            return
    
    print(-1)

if __name__ == "__main__":
    main()