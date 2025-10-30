MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = [0] + list(map(int, data[1:N+1]))
    
    inv_N = pow(N, MOD - 2, MOD)
    base = (N + 1) * inv_N % MOD
    
    E = 0
    power = 1  # base^{y-1}
    for y in range(1, N+1):
        p_y = power * inv_N % MOD
        E = (E + A[y] * p_y) % MOD
        power = power * base % MOD
    
    print(E)

if __name__ == '__main__':
    main()