# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A = list(map(int, data[2:2+T]))
    
    row_counts = [0] * N
    col_counts = [0] * N
    diag1 = 0
    diag2 = 0
    
    marked = set()
    
    for i in range(T):
        x = A[i]
        x -= 1  # zero-based
        row = x // N
        col = x % N
        
        row_counts[row] +=1
        if row_counts[row] == N:
            print(i+1)
            return
        
        col_counts[col] +=1
        if col_counts[col] == N:
            print(i+1)
            return
        
        if row == col:
            diag1 +=1
            if diag1 == N:
                print(i+1)
                return
        if row + col == N-1:
            diag2 +=1
            if diag2 == N:
                print(i+1)
                return
    print(-1)

if __name__ == "__main__":
    main()