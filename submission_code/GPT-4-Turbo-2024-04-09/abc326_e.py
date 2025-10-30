def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Calculate the expected value
    # E = A_1 + (1/2)A_2 + (1/3)A_3 + ... + (1/N)A_N
    # E = sum((1/i) * A_i for i in range(1, N+1))
    
    # We need to calculate this modulo 998244353
    # We use the property that (1/i) % p = i^(p-2) % p when p is prime (Fermat's little theorem)
    
    # Precompute the inverses using Fermat's little theorem
    inv = [1] * (N + 1)
    for i in range(2, N + 1):
        inv[i] = (MOD - MOD // i) * inv[MOD % i] % MOD
    
    expected_value = 0
    for i in range(1, N + 1):
        expected_value = (expected_value + A[i-1] * inv[i]) % MOD
    
    print(expected_value)