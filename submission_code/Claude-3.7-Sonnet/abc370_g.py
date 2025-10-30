import math

def solve(N, M):
    MOD = 998244353
    memo = {}
    
    # Compute sigma mod 3 for a given integer
    def compute_sigma_mod3(n):
        if n == 1:
            return 1
        
        # Factorize n and compute sigma mod 3
        result = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                e = 0
                while n % i == 0:
                    e += 1
                    n //= i
                
                # Compute sigma(i^e) mod 3
                if i == 3:
                    sigma_power = 1
                elif i == 2:
                    sigma_power = 0 if e % 2 == 1 else 1
                elif i % 3 == 1:
                    sigma_power = (1 + e) % 3
                elif i % 3 == 2:
                    sigma_power = 0 if e % 2 == 1 else 1
                
                result = (result * sigma_power) % 3
            i += 1
        
        if n > 1:  # n is a prime
            if n == 3:
                result = (result * 1) % 3
            elif n == 2:
                result = (result * 0) % 3
            elif n % 3 == 1:
                result = (result * 2) % 3
            elif n % 3 == 2:
                result = (result * 0) % 3
        
        return result
    
    # Compute F(n, m, r)
    def compute_F(n, m, r):
        if (n, m, r) in memo:
            return memo[(n, m, r)]
        
        if m == 1:
            # Base case: count integers <= n with sigma mod 3 = r
            count = 0
            for i in range(1, min(n + 1, 10**4)):  # Adjust the limit as needed
                if compute_sigma_mod3(i) == r:
                    count += 1
            
            memo[(n, m, r)] = count
            return count
        
        # Compute the distinct j values
        j_values = []
        i = 1
        while i <= n:
            j_values.append(n // i)
            i = n // (n // i) + 1
        
        result = 0
        for j in j_values:
            # Compute the range [l, r] such that floor(n/i) = j
            l = n // (j + 1) + 1
            r = n // j
            
            # Compute count(j, s) for s = 0, 1, 2
            count = [0, 0, 0]
            for i in range(l, min(r + 1, 10**4)):  # Adjust the limit as needed
                count[compute_sigma_mod3(i)] += 1
            
            # For larger ranges, use an approximation
            if r >= 10**4:
                remaining = r - max(l, 10**4) + 1
                count[0] += remaining // 3
                count[1] += remaining // 3
                count[2] += remaining - 2 * (remaining // 3)
            
            # Update the result
            if r == 0:
                result = (result + count[0] * compute_F(j, m-1, 0)) % MOD
                result = (result + count[1] * compute_F(j, m-1, 0)) % MOD
                result = (result + count[2] * compute_F(j, m-1, 0)) % MOD
            elif r == 1:
                result = (result + count[0] * compute_F(j, m-1, 0)) % MOD
                result = (result + count[1] * compute_F(j, m-1, 1)) % MOD
                result = (result + count[2] * compute_F(j, m-1, 2)) % MOD
            else:  # r == 2
                result = (result + count[0] * compute_F(j, m-1, 0)) % MOD
                result = (result + count[1] * compute_F(j, m-1, 2)) % MOD
                result = (result + count[2] * compute_F(j, m-1, 1)) % MOD
        
        memo[(n, m, r)] = result
        return result
    
    return compute_F(N, M, 0)

# Example usage
N, M = map(int, input().split())
result = solve(N, M)
print(result)