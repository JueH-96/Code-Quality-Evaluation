def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Compute prefix XOR array Q of length N+1
    # Q[i] = A[0]^A[1]^...^A[i-1]  (Q[0] = 0)
    Q = [0]*(N+1)
    for i in range(1, N+1):
        Q[i] = Q[i-1] ^ A[i-1]

    answer = 0

    # We only need up to 32 bits because A_i <= 10^8
    for b in range(32):
        # Extract the b-th bit of each element in Q
        B = [(Q[i] >> b) & 1 for i in range(N+1)]

        # Build suffix counts of how many 1s or 0s from index i..N
        suffix_1 = [0]*(N+2)
        suffix_0 = [0]*(N+2)
        for i in reversed(range(N+1)):
            suffix_1[i] = suffix_1[i+1] + (B[i] == 1)
            suffix_0[i] = suffix_0[i+1] + (B[i] == 0)

        # Count pairs i<j for which B[i]!=B[j], ignoring the j=i+1 restriction
        sum0_1 = 0  # count of pairs (i,j) with B[i]=0, B[j]=1
        sum1_0 = 0  # count of pairs (i,j) with B[i]=1, B[j]=0
        for i in range(N+1):
            if B[i] == 0:
                sum0_1 += suffix_1[i+1]  # all j>i with B[j]=1
            else:
                sum1_0 += suffix_0[i+1]  # all j>i with B[j]=0
        total_diff = sum0_1 + sum1_0

        # Now subtract pairs where j = i+1 (adjacent) if they differ
        # because we only want j >= i+2
        diff_adj = 0
        for i in range(N):
            if B[i] != B[i+1]:
                diff_adj += 1

        # Number of valid pairs (j>=i+2) that differ in bit b
        pairs_count = total_diff - diff_adj

        # Contribution from bit b
        answer += (pairs_count << b)

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()