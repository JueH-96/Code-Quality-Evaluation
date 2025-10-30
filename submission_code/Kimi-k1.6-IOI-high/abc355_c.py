import sys

def main():
    n, t = map(int, sys.stdin.readline().split())
    a_list = list(map(int, sys.stdin.readline().split()))
    
    rows = [0] * (n + 1)
    cols = [0] * (n + 1)
    diag1 = 0
    diag2 = 0
    
    for idx in range(t):
        a = a_list[idx]
        r = (a - 1) // n + 1
        c = (a - 1) % n + 1
        
        # Update row
        rows[r] += 1
        if rows[r] == n:
            print(idx + 1)
            return
        
        # Update column
        cols[c] += 1
        if cols[c] == n:
            print(idx + 1)
            return
        
        # Check diagonal 1 (r == c)
        if r == c:
            diag1 += 1
            if diag1 == n:
                print(idx + 1)
                return
        
        # Check diagonal 2 (r + c == n + 1)
        if r + c == n + 1:
            diag2 += 1
            if diag2 == n:
                print(idx + 1)
                return
    
    print(-1)

if __name__ == "__main__":
    main()