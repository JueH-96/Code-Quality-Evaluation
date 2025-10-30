def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read input: first token is N, next N tokens are the list A, and the last token is the string S.
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    S = data[1+N]
    
    # Precompute prefix counts for positions with letter 'M'.
    # mprefix[d][i] will store the number of indices < i (0-indexed) with letter 'M' and A value d.
    mprefix = [[0]*(N+1) for _ in range(3)]
    for i in range(N):
        # Copy previous counts.
        mprefix[0][i+1] = mprefix[0][i]
        mprefix[1][i+1] = mprefix[1][i]
        mprefix[2][i+1] = mprefix[2][i]
        if S[i] == 'M':
            d = A[i]
            mprefix[d][i+1] += 1

    # Precompute suffix counts for positions with letter 'X'.
    # xsuffix[d][i] will store the number of indices from i to N-1 with letter 'X' and A value d.
    xsuffix = [[0]*(N+1) for _ in range(3)]
    for i in range(N-1, -1, -1):
        xsuffix[0][i] = xsuffix[0][i+1]
        xsuffix[1][i] = xsuffix[1][i+1]
        xsuffix[2][i] = xsuffix[2][i+1]
        if S[i] == 'X':
            d = A[i]
            xsuffix[d][i] += 1

    # Precompute the mex for all triple combinations (x, y, z) where each is in {0, 1, 2}.
    # mex(x, y, z) is the smallest non-negative integer not in {x, y, z}.
    mex_table = {}
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sset = {a, b, c}
                m = 0
                while m in sset:
                    m += 1
                mex_table[(a, b, c)] = m

    result = 0
    # We need to sum contributions for each triple (i, j, k) with i < j < k such that S[i]=='M', S[j]=='E', S[k]=='X'.
    # For each position j with letter 'E', count all M positions (with various A values) before it and
    # all X positions (with various A values) after it.
    for j in range(N):
        if S[j] != 'E':
            continue
        y = A[j]
        # Count of M positions with each A value before index j.
        m0 = mprefix[0][j]
        m1 = mprefix[1][j]
        m2 = mprefix[2][j]
        # Count of X positions with each A value after index j (i.e. from j+1 onward).
        x0 = xsuffix[0][j+1]
        x1 = xsuffix[1][j+1]
        x2 = xsuffix[2][j+1]
        
        # For every combination of x (from M positions) and z (from X positions),
        # add the contribution: (# M positions with value x) * (# X positions with value z) * mex(x, y, z)
        if m0:
            result += m0 * x0 * mex_table[(0, y, 0)]
            result += m0 * x1 * mex_table[(0, y, 1)]
            result += m0 * x2 * mex_table[(0, y, 2)]
        if m1:
            result += m1 * x0 * mex_table[(1, y, 0)]
            result += m1 * x1 * mex_table[(1, y, 1)]
            result += m1 * x2 * mex_table[(1, y, 2)]
        if m2:
            result += m2 * x0 * mex_table[(2, y, 0)]
            result += m2 * x1 * mex_table[(2, y, 1)]
            result += m2 * x2 * mex_table[(2, y, 2)]
    
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()