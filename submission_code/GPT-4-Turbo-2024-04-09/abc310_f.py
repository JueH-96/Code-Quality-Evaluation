MOD = 998244353

def modinv(a, p):
    return pow(a, p - 2, p)

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We need to find the number of ways to get a sum of exactly 10
    # using a subset of the dice.
    
    # dp[j] will be the number of ways to get a sum of exactly j
    dp = [0] * 11
    dp[0] = 1  # There's one way to get a sum of 0: use no dice.
    
    for a in A:
        # We need to update dp in reverse to prevent overwriting
        # the results of the current round when calculating.
        new_dp = dp[:]
        for j in range(11):
            if j <= a:
                # If the die can achieve this value, add all ways to get `j` using this die
                new_dp[j] = (new_dp[j] + dp[j]) % MOD
            if j > 0:
                # Add ways to get `j-1` to `j` if possible (using the die with value 1 to a)
                new_dp[j] = (new_dp[j] + dp[j-1] * min(j, a)) % MOD
        dp = new_dp
    
    # dp[10] now contains the number of ways to get exactly 10.
    total_ways = dp[10]
    
    # Total possible outcomes
    total_outcomes = 1
    for a in A:
        total_outcomes = total_outcomes * (a + 1) % MOD
    
    # We need to find total_ways / total_outcomes % MOD
    # which is total_ways * modinv(total_outcomes, MOD) % MOD
    result = total_ways * modinv(total_outcomes, MOD) % MOD
    print(result)