def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    row_marks = [0] * N
    col_marks = [0] * N
    main_diagonal_mark = 0
    anti_diagonal_mark = 0
    row_bingo_counts = 0
    col_bingo_counts = 0
    diagonal_bingo_counts = 0
    
    for turn, A_i in enumerate(A, start=1):
        row = (A_i - 1) // N
        col = (A_i - 1) % N
        
        row_marks[row] += 1
        if row_marks[row] == N:
            row_bingo_counts += 1
        
        col_marks[col] += 1
        if col_marks[col] == N:
            col_bingo_counts += 1
        
        if row == col:
            main_diagonal_mark += 1
            if main_diagonal_mark == N:
                diagonal_bingo_counts += 1
        
        if row + col == N - 1:
            anti_diagonal_mark += 1
            if anti_diagonal_mark == N:
                diagonal_bingo_counts += 1
        
        if row_bingo_counts >= 1 or col_bingo_counts >= 1 or diagonal_bingo_counts >= 1:
            print(turn)
            return
    
    print(-1)

if __name__ == "__main__":
    main()