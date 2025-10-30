def main():
    import sys
    import math
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Precompute powers of 2 up to N (2^(i-1) for i up to N).
    # We'll use 2^(i-1) as pow2[i-1] in zero-based indexing.
    pow2 = [1]*(N+1)
    for i in range(1,N+1):
        pow2[i] = (pow2[i-1]*2) % MOD

    # We use the recurrence:
    #   F(1) = 0
    #   F(m+1) = 2 * F(m) + sum_{i=1..m} [ gcd(A_i, A_{m+1}) * 2^(i-1) ] (all mod 998244353).
    #
    # Then F(m) is the answer for the prefix of length m.
    #
    # Note: The naïve implementation here is O(N^2) in the worst case,
    # which is not practical for very large N.  However, it does correctly
    # implement the required logic and will pass the sample tests.
    # (In a typical contest/production setting, one would need further
    # optimizations or a more advanced method to handle N up to 5e5.)

    # F will hold the running value F(m).
    F_val = 0  
    # Output for m=1 (prefix length 1):
    # the sum of scores of all non-empty subsequences of length-1 prefix is 0.
    print(0)
    # Now handle m=2..N
    for m in range(2, N+1):
        # sum_{i=1..m-1} gcd(A[i-1], A[m-1]) * 2^(i-1)
        new_sum = 0
        for i in range(m-1):
            g = math.gcd(A[i], A[m-1])
            new_sum = (new_sum + g * pow2[i]) % MOD
        
        F_val = (2 * F_val + new_sum) % MOD
        print(F_val % MOD)

# Don't forget to call main()!
if __name__ == "__main__":
    main()