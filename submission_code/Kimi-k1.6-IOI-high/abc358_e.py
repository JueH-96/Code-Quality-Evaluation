import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:27]))
    
    # Precompute factorial and inverse factorial modulo MOD
    fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (K + 1)
    inv_fact[K] = pow(fact[K], MOD-2, MOD)
    for i in range(K-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Generate each character's polynomial
    polys = []
    for c_i in C:
        poly = [0] * (K + 1)
        for c in range(K + 1):
            if c <= c_i:
                poly[c] = inv_fact[c]
            else:
                poly[c] = 0
        polys.append(poly)
    
    # Multiply all polynomials
    result = [0] * (K + 1)
    result[0] = 1  # Initial polynomial is 1 (empty product)
    
    for poly in polys:
        new_result = [0] * (K + 1)
        for d in range(K + 1):
            for j in range(d + 1):
                new_result[d] = (new_result[d] + result[j] * poly[d - j]) % MOD
        result = new_result
    
    # Calculate the sum for lengths 1 to K
    total = 0
    for n in range(1, K + 1):
        total = (total + result[n] * fact[n]) % MOD
    
    print(total)

if __name__ == '__main__':
    main()