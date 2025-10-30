import sys
MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())
    if N == 0:
        print(0)
        return
    
    sqrt_N = int(N**0.5) + 1

    # Generate primes up to sqrt_N using sieve
    sieve = [True] * (sqrt_N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(sqrt_N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, sqrt_N + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    dp = [1, 0]  # dp[0] is state 0, dp[1] is state 1

    for p in primes:
        if p > sqrt_N:
            continue
        
        max_e = 0
        current = 1
        while current * p <= N:
            current *= p
            max_e += 1
        
        max_e = min(max_e, M)  # since C(e_p + M-1, M-1) is 1 for e_p >= M
        
        for e_p in range(0, max_e + 1):
            c = 1
            if e_p > 0:
                c = 1
                if e_p + M - 1 >= M:
                    c = 1
                else:
                    c = 1
            
            if e_p == 0:
                new_dp0 = dp[0] * 1
                new_dp1 = dp[1] * 1
            else:
                new_dp0 = dp[0] * 1
                new_dp1 = dp[1] * 1
            
            for s in [0, 1]:
                if dp[s] == 0:
                    continue
                if e_p == 0:
                    if s == 0:
                        s_new = 0
                    else:
                        s_new = 1
                else:
                    if p % 3 == 2:
                        if s == 0:
                            s_new = 1
                        else:
                            s_new = 1
                    else:
                        if s == 0:
                            s_new = 0
                        else:
                            s_new = 1
                
                if s_new == 0:
                    new_dp0 = (new_dp0 + dp[s] * c) % MOD
                else:
                    new_dp1 = (new_dp1 + dp[s] * c) % MOD
            
            dp = [new_dp0, new_dp1]

    # Process primes p > sqrt_N
    for p in primes:
        if p > sqrt_N:
            continue
        
        max_e = 0
        current = 1
        while current * p <= N:
            current *= p
            max_e += 1
        
        for e_p in [0, 1]:
            if e_p == 0:
                new_dp0 = dp[0] * 1
                new_dp1 = dp[1] * 1
            else:
                new_dp0 = dp[0] * 1
                new_dp1 = dp[1] * 1
            
            for s in [0, 1]:
                if dp[s] == 0:
                    continue
                if e_p == 0:
                    if s == 0:
                        s_new = 0
                    else:
                        s_new = 1
                else:
                    if p % 3 == 2:
                        if s == 0:
                            s_new = 1
                        else:
                            s_new = 1
                    else:
                        if s == 0:
                            s_new = 0
                        else:
                            s_new = 1
                
                if s_new == 0:
                    new_dp0 = (new_dp0 + dp[s] * 1) % MOD
                else:
                    new_dp1 = (new_dp1 + dp[s] * 1) % MOD
            
            dp = [new_dp0, new_dp1]

    print(dp[1] % MOD)

if __name__ == "__main__":
    main()