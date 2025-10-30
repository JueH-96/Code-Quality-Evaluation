def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    S = input[ptr].strip()

    # Compute prefix_M
    current = [0, 0, 0]
    prefix_M = [current.copy()]
    for i in range(N):
        if S[i] == 'M':
            a = A[i]
            new_current = current.copy()
            new_current[a] += 1
            current = new_current
        prefix_M.append(current.copy())

    # Compute suffix_X
    current = [0, 0, 0]
    suffix_X = [[0]*3 for _ in range(N)]
    for j in range(N-1, -1, -1):
        suffix_X[j] = current.copy()
        if S[j] == 'X':
            a = A[j]
            new_current = current.copy()
            new_current[a] += 1
            current = new_current

    # Precompute mex table
    mex_table = [[[0 for _ in range(3)] for __ in range(3)] for ___ in range(3)]
    for a in range(3):
        for e in range(3):
            for b in range(3):
                s = {a, e, b}
                m = 0
                while m in s:
                    m += 1
                mex_table[a][e][b] = m

    # Calculate the total sum
    sum_total = 0
    for j in range(N):
        if S[j] == 'E':
            e_val = A[j]
            m = prefix_M[j]  # [m0, m1, m2]
            x = suffix_X[j]  # [x0, x1, x2]
            for a in range(3):
                ma = m[a]
                for b in range(3):
                    xb = x[b]
                    sum_total += ma * xb * mex_table[a][e_val][b]

    print(sum_total)

if __name__ == '__main__':
    main()