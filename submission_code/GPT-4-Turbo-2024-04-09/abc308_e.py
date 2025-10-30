def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    S = data[N+1]
    
    # We need to find all (i, j, k) such that S_iS_jS_k == "MEX"
    # and then calculate the sum of mex(A_i, A_j, A_k) for these tuples.
    
    # To efficiently find these tuples, we can use prefix counts.
    # We will count how many 'M', 'E', and 'X' have appeared up to each index.
    
    # Prefix counts for 'M', 'E', 'X'
    count_M = [0] * (N + 1)
    count_E = [0] * (N + 1)
    count_X = [0] * (N + 1)
    
    for i in range(N):
        count_M[i+1] = count_M[i] + (1 if S[i] == 'M' else 0)
        count_E[i+1] = count_E[i] + (1 if S[i] == 'E' else 0)
        count_X[i+1] = count_X[i] + (1 if S[i] == 'X' else 0)
    
    # To calculate mex efficiently, we can use a small function
    def mex(x, y, z):
        seen = {x, y, z}
        for m in range(3):
            if m not in seen:
                return m
        return 3  # This case will never happen as we only have 0, 1, 2 in A
    
    result = 0
    
    # We need to find all (i, j, k) such that 1 <= i < j < k <= N and S_iS_jS_k == "MEX"
    # We can iterate over j and find suitable i and k
    for j in range(2, N):
        if S[j] == 'E':
            # We need S_i = 'M' and S_k = 'X'
            # Count how many 'M' are there before j and how many 'X' are there after j
            count_M_before_j = count_M[j]
            count_X_after_j = count_X[N] - count_X[j+1]
            
            # For each valid (i, j, k) we calculate the mex
            if count_M_before_j > 0 and count_X_after_j > 0:
                for i in range(1, j):
                    if S[i-1] == 'M':
                        for k in range(j+2, N+1):
                            if S[k-1] == 'X':
                                result += mex(A[i-1], A[j], A[k-1])
    
    print(result)

if __name__ == "__main__":
    main()