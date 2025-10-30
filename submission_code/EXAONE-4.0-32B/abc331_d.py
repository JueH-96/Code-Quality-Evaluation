import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    
    pattern_lines = data[1:1+N]
    queries = data[1+N:1+N+Q]
    
    base = []
    for i in range(N):
        s = pattern_lines[i].strip()
        row = []
        for char in s:
            if char == 'B':
                row.append(1)
            else:
                row.append(0)
        base.append(row)
    
    pref = [[0] * (N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            pref[i][j] = base[i-1][j-1] + pref[i-1][j] + pref[i][j-1] - pref[i-1][j-1]
            
    total_black = pref[N][N]
    
    def F(x, y):
        if x < 0 or y < 0:
            return 0
        full_x = (x + 1) // N
        full_y = (y + 1) // N
        rem_x = (x + 1) % N
        rem_y = (y + 1) % N
        
        total = full_x * full_y * total_black
        total += full_y * pref[rem_x][N]
        total += full_x * pref[N][rem_y]
        total += pref[rem_x][rem_y]
        return total

    out_lines = []
    for i in range(Q):
        parts = queries[i].split()
        if not parts:
            continue
        A = int(parts[0])
        B = int(parts[1])
        C = int(parts[2])
        D = int(parts[3])
        res = F(C, D) - F(A-1, D) - F(C, B-1) + F(A-1, B-1)
        out_lines.append(str(res))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()