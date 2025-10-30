def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    t = int(data[1])
    arr = list(map(int, data[2:2 + t]))
    
    rows = [0] * n
    cols = [0] * n
    diag1 = 0
    diag2 = 0
    
    for i in range(t):
        num = arr[i]
        r = (num - 1) // n
        c = (num - 1) % n
        
        rows[r] += 1
        cols[c] += 1
        
        if r == c:
            diag1 += 1
        if r + c == n - 1:
            diag2 += 1
        
        if rows[r] == n or cols[c] == n or diag1 == n or diag2 == n:
            print(i + 1)
            return
    
    print(-1)

if __name__ == '__main__':
    main()