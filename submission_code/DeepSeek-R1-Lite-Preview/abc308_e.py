def mex(a, b, c):
    s = {a, b, c}
    m = 0
    while m in s:
        m += 1
    return m

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    S = input[ptr]
    
    # Initialize prefix counts
    prefix_M0 = [0] * (N + 1)
    prefix_M1 = [0] * (N + 1)
    prefix_M2 = [0] * (N + 1)
    
    for i in range(1, N + 1):
        if S[i - 1] == 'M':
            prefix_M0[i] = prefix_M0[i - 1] + (A[i - 1] == 0)
            prefix_M1[i] = prefix_M1[i - 1] + (A[i - 1] == 1)
            prefix_M2[i] = prefix_M2[i - 1] + (A[i - 1] == 2)
        else:
            prefix_M0[i] = prefix_M0[i - 1]
            prefix_M1[i] = prefix_M1[i - 1]
            prefix_M2[i] = prefix_M2[i - 1]
    
    # Initialize suffix counts
    suffix_X0 = [0] * (N + 2)
    suffix_X1 = [0] * (N + 2)
    suffix_X2 = [0] * (N + 2)
    
    for i in range(N - 1, -1, -1):
        if S[i] == 'X':
            suffix_X0[i] = suffix_X0[i + 1] + (A[i] == 0)
            suffix_X1[i] = suffix_X1[i + 1] + (A[i] == 1)
            suffix_X2[i] = suffix_X2[i + 1] + (A[i] == 2)
        else:
            suffix_X0[i] = suffix_X0[i + 1]
            suffix_X1[i] = suffix_X1[i + 1]
            suffix_X2[i] = suffix_X2[i + 1]
    
    total = 0
    for j in range(N):
        if S[j] == 'E':
            a = A[j]
            # Get prefix counts up to j
            count_M0 = prefix_M0[j]
            count_M1 = prefix_M1[j]
            count_M2 = prefix_M2[j]
            # Get suffix counts from j+1
            count_X0 = suffix_X0[j + 1]
            count_X1 = suffix_X1[j + 1]
            count_X2 = suffix_X2[j + 1]
            # Compute contribution for each combination of b and c
            # b: 0,1,2 ; c: 0,1,2
            for b in range(3):
                for c in range(3):
                    m = mex(b, a, c)
                    total += m * count_Mb * count_Xc
    print(total)

if __name__ == '__main__':
    main()