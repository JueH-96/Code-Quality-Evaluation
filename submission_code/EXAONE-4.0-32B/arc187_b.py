MOD = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    B = []
    for i in range(N):
        x = next(it)
        if x == '-1':
            B.append(-1)
        else:
            B.append(int(x))
            
    q_total = sum(1 for x in B if x == -1)
    
    U_pre = [0] * (N + 1)
    for i in range(1, N + 1):
        add = 1 if B[i - 1] == -1 else 0
        U_pre[i] = U_pre[i - 1] + add
        
    INF = 10**9
    min_fixed = [INF] * (N + 1)
    min_fixed[0] = INF
    for i in range(1, N + 1):
        if B[i - 1] == -1:
            min_fixed[i] = min_fixed[i - 1]
        else:
            min_fixed[i] = min(min_fixed[i - 1], B[i - 1])
            
    F_table = [[0] * (M + 1) for _ in range(N + 1)]
    for e in range(0, N + 1):
        arr = [0] * (M + 1)
        for x in range(1, M + 1):
            base = M - x
            arr[x] = pow(base, e, MOD)
        s_arr = [0] * (M + 1)
        for x in range(1, M + 1):
            s_arr[x] = (s_arr[x - 1] + arr[x]) % MOD
        for L in range(0, M + 1):
            F_table[e][L] = s_arr[L]
            
    ans = 0
    for i in range(0, N):
        u_i = U_pre[i]
        if B[i] != -1:
            y = B[i]
            if min_fixed[i] <= y:
                cont_i = 0
            else:
                term1 = pow(M - y, u_i, MOD)
                exp_suffix = q_total - u_i
                term2 = pow(M, exp_suffix, MOD)
                cont_i = term1 * term2 % MOD
            ans = (ans + cont_i) % MOD
        else:
            if min_fixed[i] > M:
                L = M
            else:
                L = min_fixed[i] - 1
                if L < 0:
                    L = 0
            S_val = F_table[u_i][L]
            exp_suffix = q_total - u_i - 1
            term2 = pow(M, exp_suffix, MOD)
            cont_i = S_val * term2 % MOD
            ans = (ans + cont_i) % MOD
            
    print(ans % MOD)

if __name__ == '__main__':
    main()