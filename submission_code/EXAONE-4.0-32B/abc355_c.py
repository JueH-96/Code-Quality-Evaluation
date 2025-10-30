import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    t = int(data[1])
    A = list(map(int, data[2:2+t]))
    
    row_count = [0] * n
    col_count = [0] * n
    diag1 = 0
    diag2 = 0
    
    for i in range(t):
        num = A[i]
        r = (num - 1) // n
        c = (num - 1) % n
        
        row_count[r] += 1
        if row_count[r] == n:
            print(i + 1)
            return
            
        col_count[c] += 1
        if col_count[c] == n:
            print(i + 1)
            return
            
        if r == c:
            diag1 += 1
            if diag1 == n:
                print(i + 1)
                return
                
        if r + c == n - 1:
            diag2 += 1
            if diag2 == n:
                print(i + 1)
                return
                
    print(-1)

if __name__ == "__main__":
    main()