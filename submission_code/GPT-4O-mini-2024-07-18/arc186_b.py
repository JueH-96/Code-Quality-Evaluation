def count_permutations(N, A):
    MOD = 998244353
    
    # Initialize factorials and inverse factorials
    fact = [1] * (N + 1)
    inv_fact = [1] * (N + 1)
    
    # Precompute factorials and inverse factorials
    for i in range(2, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    # Count the number of valid permutations
    count = 1
    last_position = -1
    
    for i in range(N):
        if A[i] > 0:
            # Count how many positions are available for P[i]
            available_positions = i - A[i] - (last_position + 1)
            if available_positions < 0:
                return 0
            count = count * fact[available_positions] % MOD
        last_position = i
    
    return count

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N + 1]))
    
    result = count_permutations(N, A)
    print(result)

if __name__ == "__main__":
    main()