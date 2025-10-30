def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    K = int(data[0])
    C = list(map(int, data[1:]))
    
    MOD = 998244353
    
    # We need to calculate the number of valid strings of length 1 to K
    # where each character can appear between 0 and C[i] times.
    
    # dp[length][used] will be the number of ways to form a string of length `length`
    # using exactly `used` characters.
    dp = [[0] * (K + 1) for _ in range(K + 1)]
    dp[0][0] = 1  # 1 way to make an empty string with 0 characters
    
    # Precompute factorials and inverse factorials for combinations
    fact = [1] * (K + 1)
    ifact = [1] * (K + 1)
    for i in range(2, K + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    ifact[K] = pow(fact[K], MOD - 2, MOD)
    for i in range(K - 1, 0, -1):
        ifact[i] = ifact[i + 1] * (i + 1) % MOD
    
    def comb(n, k):
        if n < k or k < 0:
            return 0
        return fact[n] * ifact[k] % MOD * ifact[n - k] % MOD
    
    # Iterate over each character possibility
    for c in C:
        if c == 0:
            continue
        # We need to update the dp table considering we can use this character up to `c` times
        for length in range(K, 0, -1):  # Go backwards to prevent overwriting
            for used in range(K, -1, -1):  # Also backwards
                # Add ways to extend strings using this character from 1 to min(c, length) times
                for add in range(1, min(c, length - used) + 1):
                    dp[length][used + add] = (dp[length][used + add] + dp[length - add][used] * comb(used + add, add)) % MOD
    
    # Sum up all valid dp[length][used] where 1 <= length <= K and used <= length
    result = 0
    for length in range(1, K + 1):
        for used in range(1, length + 1):
            result = (result + dp[length][used]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()