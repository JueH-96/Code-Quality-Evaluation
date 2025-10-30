def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))
    
    # We need to find the sum of scores of all non-empty subsequences of A.
    # A subsequence's score is (XOR of elements)^K if its length is a multiple of M, otherwise 0.
    
    # To efficiently calculate this, we can use dynamic programming with a twist of combinatorial counting.
    
    # dp[x][r] will store the number of ways to get a XOR sum of `x` with a subsequence of length `r mod M`.
    # We only need to keep track of lengths modulo M.
    
    # Maximum possible XOR value for 20-bit numbers is less than 2^20
    max_xor = (1 << 20)  # 1048576
    
    # Initialize dp array
    dp = [[0] * M for _ in range(max_xor)]
    dp[0][0] = 1  # There's one way to have an XOR sum of 0 with an empty subsequence
    
    # Update dp table based on elements in A
    for num in A:
        # We need to iterate backwards to prevent overwriting issues during updates
        new_dp = [row[:] for row in dp]  # Make a copy of current dp state
        for xor_sum in range(max_xor):
            for r in range(M):
                new_r = (r + 1) % M
                new_xor = xor_sum ^ num
                new_dp[new_xor][new_r] = (new_dp[new_xor][new_r] + dp[xor_sum][r]) % MOD
        
        dp = new_dp
    
    # Now calculate the result based on dp values
    result = 0
    for xor_sum in range(max_xor):
        for r in range(M):
            if r == 0 and xor_sum != 0:  # We need non-empty subsequences
                # Calculate (xor_sum ^ K) % MOD
                xor_power = pow(xor_sum, K, MOD)
                result = (result + xor_power * dp[xor_sum][r]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()