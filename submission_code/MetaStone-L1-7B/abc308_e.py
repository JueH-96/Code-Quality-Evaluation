def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    S = input[ptr]
    ptr += 1

    # Precompute mex_table[x][a][y] for all x, a, y
    mex_table = [[[0 for _ in range(3)] for __ in range(3)] for ___ in range(3)]
    for x in range(3):
        for a in range(3):
            for y in range(3):
                s = {x, a, y}
                mex = 0
                while mex in s:
                    mex += 1
                mex_table[x][a][y] = mex

    # Compute prefix_m: prefix_m[j] is the counts of M's before j with A_i = 0,1,2
    prefix_m = [[0]*3 for _ in range(N+1)]
    for j in range(N):
        prefix_m[j+1] = prefix_m[j].copy()
        if S[j] == 'M':
            a = A[j]
            prefix_m[j+1][a] += 1

    # Compute suffix_x: suffix_x[j] is the counts of X's after j with A_k = 0,1,2
    suffix_x = [[0]*3 for _ in range(N+1)]
    for j in range(N-1, -1, -1):
        suffix_x[j] = suffix_x[j+1].copy()
        if S[j] == 'X':
            a = A[j]
            suffix_x[j][a] += 1

    total = 0
    for j in range(N):
        if S[j] == 'E':
            a = A[j]
            m_counts = prefix_m[j]
            x_counts = suffix_x[j]
            for x in range(3):
                for y in range(3):
                    total += m_counts[x] * x_counts[y] * mex_table[x][a][y]
    print(total)

if __name__ == '__main__':
    main()