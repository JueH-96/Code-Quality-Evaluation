def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    S = input[ptr]
    
    # Precompute mex table
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                mask = 0
                mask |= 1 << a
                mask |= 1 << b
                mask |= 1 << c
                mex = 0
                while (mask & (1 << mex)) != 0:
                    mex +=1
                mex_table[a][b][c] = mex
    
    # Precompute prefix sums for M's
    m0 = [0]*(N+1)
    m1 = [0]*(N+1)
    m2 = [0]*(N+1)
    for i in range(N):
        m0[i+1] = m0[i] + (1 if S[i] == 'M' and A[i] ==0 else 0)
        m1[i+1] = m1[i] + (1 if S[i] == 'M' and A[i] ==1 else 0)
        m2[i+1] = m2[i] + (1 if S[i] == 'M' and A[i] ==2 else 0)
    
    # Precompute suffix sums for X's
    x0 = [0]*(N+2)
    x1 = [0]*(N+2)
    x2 = [0]*(N+2)
    for i in range(N-1, -1, -1):
        x0[i] = x0[i+1] + (1 if S[i] == 'X' and A[i] ==0 else 0)
        x1[i] = x1[i+1] + (1 if S[i] == 'X' and A[i] ==1 else 0)
        x2[i] = x2[i+1] + (1 if S[i] == 'X' and A[i] ==2 else 0)
    
    total = 0
    for j in range(N):
        if S[j] == 'E':
            b = A[j]
            mc0 = m0[j]
            mc1 = m1[j]
            mc2 = m2[j]
            xc0 = x0[j+1]
            xc1 = x1[j+1]
            xc2 = x2[j+1]
            contrib = 0
            for a in range(3):
                for c in range(3):
                    ma = [mc0, mc1, mc2][a]
                    xc = [xc0, xc1, xc2][c]
                    contrib += ma * xc * mex_table[a][b][c]
            total += contrib
    print(total)

if __name__ == "__main__":
    main()