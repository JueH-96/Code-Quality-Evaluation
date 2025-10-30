import sys

def main():
    n, t = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    row = [0] * n
    col = [0] * n
    diag1 = 0
    diag2 = 0
    
    for turn in range(t):
        x = a[turn]
        idx = x - 1
        i = idx // n
        j = idx % n
        
        # Update row count
        row[i] += 1
        if row[i] == n:
            print(turn + 1)
            return
        
        # Update column count
        col[j] += 1
        if col[j] == n:
            print(turn + 1)
            return
        
        # Check diagonals
        if i == j:
            diag1 += 1
            if diag1 == n:
                print(turn + 1)
                return
        if i + j == n - 1:
            diag2 += 1
            if diag2 == n:
                print(turn + 1)
                return
    
    print(-1)

if __name__ == "__main__":
    main()