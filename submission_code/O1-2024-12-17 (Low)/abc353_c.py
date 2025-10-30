def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # 1) Compute the sum of all A_i
    S = sum(A)
    
    # 2) Compute the total of (A_i + A_j) over all i < j (without modulo)
    #    This is (N-1) * S
    total_sum = (N - 1) * S
    
    # 3) We need to find how many pairs (i < j) satisfy A_i + A_j >= 10^8.
    #    It is easier to find how many pairs have A_i + A_j < 10^8, then subtract from total pairs.
    
    A.sort()
    K = 10**8
    
    # Count of pairs with sum < K
    c = 0
    
    i = 0
    j = N - 1
    
    # Two-pointer approach to count # pairs with sum < K
    while i < j:
        if A[i] + A[j] < K:
            c += (j - i)  # all pairs (i, x) with x in [i+1..j]
            i += 1
        else:
            j -= 1
    
    # Total number of pairs
    M = N * (N - 1) // 2
    
    # Number of pairs with sum >= K
    big_pairs = M - c
    
    # The desired sum = sum of (A_i + A_j) mod 10^8 over i<j
    # = total_sum - K * (number of pairs whose sum >= K)
    answer = total_sum - K * big_pairs
    
    print(answer)

# Do not forget to call main!!
main()