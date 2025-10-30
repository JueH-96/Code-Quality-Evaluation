MOD = 998244353

def read_input():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    return N, A

def main():
    N, A = read_input()
    
    # dp[i] will store the number of Polish sequences of length i that are lexicographically less than or equal to A[0:i]
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: empty sequence is trivially Polish and less than any non-empty prefix of A
    
    # sum_dp[i] will be the sum of dp[0] through dp[i]
    sum_dp = [0] * (N + 1)
    sum_dp[0] = dp[0]
    
    for i in range(1, N + 1):
        if A[i-1] == 0:
            # Only the sequence (0) can be considered if A[i-1] is 0
            dp[i] = dp[i-1] if i == 1 else 0
        else:
            # Calculate the number of valid Polish sequences ending at position i
            # We need to consider all possible lengths of the first block
            start = i - 1
            count = 0
            while start >= 0:
                length = i - start
                if A[start] > 0 and length % (A[start] + 1) == 0:
                    num_blocks = length // (A[start] + 1)
                    if num_blocks == A[start]:
                        if start == 0:
                            count = (count + dp[start]) % MOD
                        else:
                            count = (count + dp[start] - dp[start - 1]) % MOD
                start -= 1
            
            dp[i] = (dp[i-1] + count) % MOD
        
        sum_dp[i] = (sum_dp[i-1] + dp[i]) % MOD
    
    # The result is the number of Polish sequences of length N that are lexicographically less than or equal to A
    print(dp[N])

if __name__ == "__main__":
    main()