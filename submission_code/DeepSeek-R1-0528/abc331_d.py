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
    
    total_black = 0
    pref = []
    
    if N > 0:
        pref = [[0] * N for _ in range(N)]
        for i in range(N):
            line = pattern_lines[i].strip()
            for j in range(N):
                val = 1 if line[j] == 'B' else 0
                if i == 0 and j == 0:
                    pref[i][j] = val
                elif i == 0:
                    pref[i][j] = pref[i][j-1] + val
                elif j == 0:
                    pref[i][j] = pref[i-1][j] + val
                else:
                    pref[i][j] = pref[i-1][j] + pref[i][j-1] - pref[i-1][j-1] + val
        total_black = pref[N-1][N-1]
    
    def F(x, y):
        if x == 0 or y == 0:
            return 0
        full_rows = x // N
        rem_rows = x % N
        full_cols = y // N
        rem_cols = y % N
        
        total = full_rows * full_cols * total_black
        
        if rem_rows > 0:
            total += full_cols * pref[rem_rows-1][N-1]
        if rem_cols > 0:
            total += full_rows * pref[N-1][rem_cols-1]
        if rem_rows > 0 and rem_cols > 0:
            total += pref[rem_rows-1][rem_cols-1]
            
        return total

    output_lines = []
    for i in range(Q):
        parts = queries[i].split()
        A = int(parts[0])
        B = int(parts[1])
        C = int(parts[2])
        D = int(parts[3])
        res = F(C+1, D+1) - F(A, D+1) - F(C+1, B) + F(A, B)
        output_lines.append(str(res))
    
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()