MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    
    max_d = N
    sum_powers = [[0] * (M + 2) for _ in range(max_d + 1)]
    for d in range(max_d + 1):
        for x in range(1, M + 1):
            if d == 0:
                sum_powers[d][x] = x % MOD
            else:
                sum_powers[d][x] = (sum_powers[d][x-1] + pow(x, d, MOD)) % MOD
    
    min_F = [float('inf')] * N
    u_count = [0] * N
    current_min = float('inf')
    current_u = 0
    for s in range(N):
        if s == 0:
            min_F[s] = float('inf')
            u_count[s] = 0
        else:
            i = s - 1
            current_B = B[i]
            if current_B != -1:
                if current_B < current_min:
                    current_min = current_B
            else:
                current_u += 1
            min_F[s] = current_min
            u_count[s] = current_u
    
    pow_M = [1] * (N + 2)
    for i in range(1, N + 2):
        pow_M[i] = (pow_M[i-1] * M) % MOD
    
    total = 0
    for s in range(N):
        f_filtered = [x for x in B[:s] if x != -1]
        if f_filtered:
            min_f = min(f_filtered)
        else:
            min_f = float('inf')
        max_v = min_f - 1 if min_f != float('inf') else M
        
        a = M - max_v
        b = M - 1
        
        u = u_count[s]
        
        if u == 0:
            sum_x = max_v % MOD
        else:
            if a > b:
                sum_x = 0
            else:
                if a == 1:
                    sum_a_minus_1 = 0
                else:
                    sum_a_minus_1 = sum_powers[u][a - 1] if a - 1 >= 1 else 0
                sum_x = (sum_powers[u][b] - sum_a_minus_1) % MOD
        
        k = (N - 1) - s
        pow_k = pow_M[k] if k >= 0 else 0
        term = (sum_x * pow_k) % MOD
        total = (total + term) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()