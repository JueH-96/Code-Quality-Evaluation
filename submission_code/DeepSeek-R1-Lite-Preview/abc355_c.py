def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    row_counts = [0] * (N + 1)
    col_counts = [0] * (N + 1)
    diag1 = 0
    diag2 = 0
    
    for turn in range(1, T + 1):
        a = A[turn - 1]
        row = (a - 1) // N + 1
        col = (a - 1) % N + 1
        
        row_counts[row] += 1
        if row_counts[row] == N:
            print(turn)
            return
        
        col_counts[col] += 1
        if col_counts[col] == N:
            print(turn)
            return
        
        if row == col:
            diag1 += 1
            if diag1 == N:
                print(turn)
                return
        
        if row + col == N + 1:
            diag2 += 1
            if diag2 == N:
                print(turn)
                return
    
    print(-1)

if __name__ == '__main__':
    main()