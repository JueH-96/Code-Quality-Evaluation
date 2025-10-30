import sys

def main():
    n, t = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    row_counts = [0] * (n + 1)
    col_counts = [0] * (n + 1)
    left_diag = 0
    right_diag = 0
    
    for turn in range(t):
        a = A[turn]
        a1 = a - 1
        row = a1 // n + 1
        col = a1 % n + 1
        
        row_counts[row] += 1
        if row_counts[row] == n:
            print(turn + 1)
            return
        
        col_counts[col] += 1
        if col_counts[col] == n:
            print(turn + 1)
            return
        
        if row == col:
            left_diag += 1
            if left_diag == n:
                print(turn + 1)
                return
        
        if row + col == n + 1:
            right_diag += 1
            if right_diag == n:
                print(turn + 1)
                return
    
    print(-1)

if __name__ == "__main__":
    main()