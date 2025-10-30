MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if not A:
        for _ in range(N):
            print(0)
        return
    
    max_a = max(A)
    max_phi = max(max_a, 1)
    
    # Precompute Euler's totient function phi using sieve
    phi = list(range(max_phi + 1))
    for p in range(2, max_phi + 1):
        if phi[p] == p:  # p is prime
            for m in range(p, max_phi + 1, p):
                phi[m] = phi[m] // p * (p - 1)
    
    # Precompute divisors for each number using sieve
    max_div = max_phi
    divisors = [[] for _ in range(max_div + 1)]
    for i in range(1, max_div + 1):
        for j in range(i, max_div + 1, i):
            divisors[j].append(i)
    
    # Precompute powers of 2 modulo MOD up to N
    max_pow = N
    pow2 = [1] * (max_pow + 1)
    for i in range(1, max_pow + 1):
        pow2[i] = pow2[i-1] * 2 % MOD
    
    # Initialize the frequency array for divisors
    max_freq = max_phi
    f = [0] * (max_freq + 2)  # Using +2 to avoid possible issues
    
    S_prev = 0
    for m in range(1, N + 1):
        x = A[m-1]
        current_divisors = divisors[x]
        C = 0
        for y in current_divisors:
            C = (C + phi[y] * f[y]) % MOD
        
        S_current = (2 * S_prev + C) % MOD
        # Update the frequency array
        add_val = pow2[m-1]
        for y in current_divisors:
            f[y] = (f[y] + add_val) % MOD
        
        print(S_current)
        S_prev = S_current

if __name__ == "__main__":
    main()