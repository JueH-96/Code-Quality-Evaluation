import sys

def main():
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    max_num = 100000
    
    # Precompute divisors for each number up to max_num
    divisors = [[] for _ in range(max_num + 1)]
    for d in range(1, max_num + 1):
        for multiple in range(d, max_num + 1, d):
            divisors[multiple].append(d)
    
    # Precompute Möbius function mu
    is_prime = [True] * (max_num + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(max_num**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p*p, max_num + 1, p):
                is_prime[multiple] = False
    primes = [p for p in range(2, max_num + 1) if is_prime[p]]
    
    mu = [1] * (max_num + 1)
    for p in primes:
        for multiple in range(p, max_num + 1, p):
            mu[multiple] *= -1
        for multiple in range(p*p, max_num + 1, p*p):
            mu[multiple] = 0
    
    # Precompute f array where f[m] = sum_{d | m} d * mu(m//d)
    f = [0] * (max_num + 1)
    for m in range(1, max_num + 1):
        total = 0
        for d in divisors[m]:
            k = m // d
            total += d * mu[k]
        f[m] = total
    
    # Precompute powers of 2 modulo MOD
    pow2 = [1] * N
    for i in range(1, N):
        pow2[i] = (pow2[i-1] * 2) % MOD
    
    # Initialize S array and previous S value
    S = [0] * (max_num + 1)
    prev_S = 0
    
    for m in range(1, N + 1):
        x = A[m-1]
        divisors_x = divisors[x]
        
        G = 0
        for d in divisors_x:
            G += f[d] * S[d]
        G %= MOD
        
        current_S = (2 * prev_S + G) % MOD
        print(current_S)
        
        add_val = pow2[m-1]
        for d in divisors_x:
            S[d] = (S[d] + add_val) % MOD
        
        prev_S = current_S

if __name__ == "__main__":
    main()