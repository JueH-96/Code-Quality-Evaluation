MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:27]))  # Read exactly 26 values
    
    # Precompute factorials and inverse factorials up to K
    max_n = K
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    # Initialize the product polynomial
    product = [0] * (K + 1)
    product[0] = 1  # x^0 term
    
    for i in range(26):
        c_i = C[i]
        # Create the EGF for this letter
        egf = [0] * (c_i + 1)
        for c in range(c_i + 1):
            egf[c] = inv_fact[c]
        
        # Multiply the current product by this EGF
        new_product = [0] * (K + 1)
        for l in range(K + 1):
            if product[l] == 0:
                continue
            for c in range(c_i + 1):
                if l + c > K:
                    continue
                new_product[l + c] = (new_product[l + c] + product[l] * egf[c]) % MOD
        product = new_product
    
    # Calculate the total number of valid strings
    total = 0
    for L in range(1, K + 1):
        total = (total + product[L] * fact[L]) % MOD
    
    print(total)

if __name__ == '__main__':
    main()