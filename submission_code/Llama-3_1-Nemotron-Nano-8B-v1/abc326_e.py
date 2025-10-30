MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Compute suffix sum of A (sum_a[x] is sum of A[y] for y from x+1 to N)
    sum_a = [0] * (N + 2)  # sum_a[x] for x in 0..N
    for x in range(N-1, -1, -1):
        sum_a[x] = (sum_a[x+1] + A[x]) % MOD
    
    inv_N = pow(N, MOD-2, MOD)
    
    sum_e = 0
    # Process x from N-1 down to 0
    for x in reversed(range(N)):
        numerator = (sum_a[x] + sum_e) % MOD
        E_x = numerator * inv_N % MOD
        sum_e = (sum_e + E_x) % MOD
    
    print(sum_e)

if __name__ == "__main__":
    main()