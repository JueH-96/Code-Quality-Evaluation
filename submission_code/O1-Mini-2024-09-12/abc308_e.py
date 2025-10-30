# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    S = N_and_rest[N+1]
    
    # Initialize prefix counts for 'M's
    prefix_M0 = [0] * (N+1)
    prefix_M1 = [0] * (N+1)
    prefix_M2 = [0] * (N+1)
    
    for j in range(1, N+1):
        prefix_M0[j] = prefix_M0[j-1] + (1 if S[j-1] == 'M' and A[j-1] == 0 else 0)
        prefix_M1[j] = prefix_M1[j-1] + (1 if S[j-1] == 'M' and A[j-1] == 1 else 0)
        prefix_M2[j] = prefix_M2[j-1] + (1 if S[j-1] == 'M' and A[j-1] == 2 else 0)
    
    # Initialize suffix counts for 'X's
    suffix_X0 = [0] * (N+1)
    suffix_X1 = [0] * (N+1)
    suffix_X2 = [0] * (N+1)
    
    for j in range(N-1, -1, -1):
        suffix_X0[j] = suffix_X0[j+1] + (1 if S[j] == 'X' and A[j] == 0 else 0)
        suffix_X1[j] = suffix_X1[j+1] + (1 if S[j] == 'X' and A[j] == 1 else 0)
        suffix_X2[j] = suffix_X2[j+1] + (1 if S[j] == 'X' and A[j] == 2 else 0)
    
    # Precompute mex table
    mex_table = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                s = {a, b, c}
                m = 0
                while m in s:
                    m +=1
                mex_table[a][b][c] = m
    
    total = 0
    for j in range(N):
        if S[j] == 'E':
            b = A[j]
            # Counts of 'M's before j
            m0 = prefix_M0[j]
            m1 = prefix_M1[j]
            m2 = prefix_M2[j]
            # Counts of 'X's after j
            x0 = suffix_X0[j+1]
            x1 = suffix_X1[j+1]
            x2 = suffix_X2[j+1]
            counts_M = [m0, m1, m2]
            counts_X = [x0, x1, x2]
            for a in range(3):
                count_M_a = counts_M[a]
                if count_M_a == 0:
                    continue
                for c in range(3):
                    count_X_c = counts_X[c]
                    if count_X_c == 0:
                        continue
                    m = mex_table[a][b][c]
                    total += m * count_M_a * count_X_c
    print(total)

if __name__ == "__main__":
    main()