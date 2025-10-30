import sys

def main():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    
    A = [int(c) for c in S_str]
    
    # dp0[i] stores the sum of sequence V starting at index i, where V_i = 0.
    # dp1[i] stores the sum of sequence V starting at index i, where V_i = 1.
    # Sequence V is defined as: V_k = V_{k-1} NAND A[k] for k > i.
    # The sum is V_i + V_{i+1} + ... + V_{N-1}.
    
    dp0 = [0] * N
    dp1 = [0] * N

    # Base cases for i = N-1 (last element)
    dp0[N-1] = 0  # If sequence is just '0', sum is 0.
    dp1[N-1] = 1  # If sequence is just '1', sum is 1.

    # Iterate backwards from i = N-2 down to 0
    for i in range(N - 2, -1, -1):
        # Calculate dp0[i]: Current term V_i is 0. Sum is 0 + rest_of_sum.
        # Next term V_{i+1} = V_i NAND A[i+1] = 0 NAND A[i+1] = 1.
        # The rest_of_sum starts with V_{i+1}=1 using A[i+1], A[i+2], ..., A[N-1].
        # This corresponds to dp1[i+1].
        dp0[i] = 0 + dp1[i+1]

        # Calculate dp1[i]: Current term V_i is 1. Sum is 1 + rest_of_sum.
        # Next term V_{i+1} = V_i NAND A[i+1] = 1 NAND A[i+1].
        # If A[i+1] == 0, V_{i+1} = 1 NAND 0 = 1. Rest of sum is dp1[i+1].
        # If A[i+1] == 1, V_{i+1} = 1 NAND 1 = 0. Rest of sum is dp0[i+1].
        if A[i+1] == 0:
            dp1[i] = 1 + dp1[i+1]
        else: # A[i+1] == 1
            dp1[i] = 1 + dp0[i+1]
            
    total_sum = 0
    # The problem asks for sum_{0<=i<=j<=N-1} f(i,j) using 0-indexing.
    # This is sum_{i=0}^{N-1} ( sum_{j=i}^{N-1} f(i,j) )
    # For each starting index i:
    #   The sequence f(i,i), f(i,i+1)... starts with f(i,i) = A[i].
    #   So if A[i] is 0, this sum is dp0[i].
    #   If A[i] is 1, this sum is dp1[i].
    for i in range(N):
        if A[i] == 0:
            total_sum += dp0[i]
        else: # A[i] == 1
            total_sum += dp1[i]
            
    print(total_sum)

if __name__ == '__main__':
    main()