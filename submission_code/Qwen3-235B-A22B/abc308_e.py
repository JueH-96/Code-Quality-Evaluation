import sys

def main():
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    s = sys.stdin.readline().strip()
    
    # Precompute mex table
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s_set = {a, b, c}
                m = 0
                while m in s_set:
                    m += 1
                mex_table[a][b][c] = m
    
    # Compute prefix counts for 'M'
    prefix = [[0]*3 for _ in range(n+1)]
    for i in range(n):
        prefix[i+1][0] = prefix[i][0]
        prefix[i+1][1] = prefix[i][1]
        prefix[i+1][2] = prefix[i][2]
        if s[i] == 'M':
            a_val = A[i]
            prefix[i+1][a_val] += 1
    
    # Compute suffix counts for 'X'
    suffix = [[0]*3 for _ in range(n+2)]
    for i in range(n-1, -1, -1):
        suffix[i][0] = suffix[i+1][0]
        suffix[i][1] = suffix[i+1][1]
        suffix[i][2] = suffix[i+1][2]
        if s[i] == 'X':
            a_val = A[i]
            suffix[i][a_val] += 1
    
    total = 0
    for j in range(n):
        if s[j] != 'E':
            continue
        b = A[j]
        m_counts = prefix[j]
        x_counts = suffix[j+1]
        for a in range(3):
            ma = m_counts[a]
            if ma == 0:
                continue
            for c in range(3):
                xc = x_counts[c]
                if xc == 0:
                    continue
                total += ma * xc * mex_table[a][b][c]
    
    print(total)

if __name__ == "__main__":
    main()