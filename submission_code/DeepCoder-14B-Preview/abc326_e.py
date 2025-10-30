MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Compute sum_A[x] = sum of A[y] for y > x (y is 1-based in the problem, but 0-based here)
    sum_A = [0] * (N + 2)  # sum_A[x] is sum of A[x], A[x+1], ..., A[N-1]
    for x in range(N-1, -1, -1):
        sum_A[x] = (sum_A[x+1] + A[x]) % MOD
    
    inv_N = pow(N, MOD-2, MOD)
    
    current_sum_E = 0
    E = [0] * (N + 2)  # E[x] for x from 0 to N
    
    for x in range(N, -1, -1):
        if x == N:
            E[x] = 0
        else:
            E[x] = (sum_A[x] + current_sum_E) * inv_N % MOD
        current_sum_E = (current_sum_E + E[x]) % MOD
    
    print(E[0] % MOD)

if __name__ == "__main__":
    main()