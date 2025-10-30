def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    A = list(map(int, input[idx].split()))
    idx += 1
    S = input[idx]
    
    # Precompute prefix_m: counts of M's before each index
    prefix_m = [(0, 0, 0)] * (N + 1)
    for j in range(N):
        if S[j] == 'M':
            a = A[j]
            prev = list(prefix_m[j])
            prev[a] += 1
            prefix_m[j+1] = tuple(prev)
        else:
            prefix_m[j+1] = prefix_m[j]
    
    # Precompute suffix_x: counts of X's after each index
    suffix_x = [(0, 0, 0)] * (N + 2)
    for j in range(N-1, -1, -1):
        if S[j] == 'X':
            a = A[j]
            prev = list(suffix_x[j+1])
            prev[a] += 1
            suffix_x[j] = tuple(prev)
        else:
            suffix_x[j] = suffix_x[j+1]
    
    total = 0
    for j in range(N):
        if S[j] == 'E':
            a_e = A[j]
            # M's before j: prefix_m[j]
            m0, m1, m2 = prefix_m[j]
            # X's after j: suffix_x[j+1]
            x0, x1, x2 = suffix_x[j+1]
            
            # Iterate all possible a and c
            for a in [0, 1, 2]:
                if a == 0:
                    m = m0
                elif a == 1:
                    m = m1
                else:
                    m = m2
                
                for c in [0, 1, 2]:
                    if c == 0:
                        x = x0
                    elif c == 1:
                        x = x1
                    else:
                        x = x2
                    
                    # Compute mex(a, a_e, c)
                    mex_val = 0
                    while mex_val in {a, a_e, c}:
                        mex_val += 1
                    total += m * x * mex_val
    print(total)

if __name__ == "__main__":
    main()