MOD = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    A_list = [int(next(it)) for _ in range(n-1)]
    
    B = [0] * (n+1)
    for i in range(2, n+1):
        B[i] = A_list[i-2]
        
    max_n = n
    fac = [1] * (max_n+1)
    for i in range(1, max_n+1):
        fac[i] = fac[i-1] * i % MOD
        
    inv = [0] * (max_n+1)
    for i in range(1, max_n+1):
        inv[i] = pow(i, MOD-2, MOD)
        
    arr1 = [0] * (n+1)
    for i in range(2, n+1):
        arr1[i] = B[i] * inv[i] % MOD
        
    prefix_arr1 = [0] * (n+1)
    for i in range(2, n+1):
        prefix_arr1[i] = (prefix_arr1[i-1] + arr1[i]) % MOD
        
    T_arr = [0] * (n+1)
    for i in range(2, n+1):
        temp = B[i] * fac[i-1] % MOD
        temp = temp * fac[n - i] % MOD
        temp = temp * (i - 1) % MOD
        T_arr[i] = temp
        
    prefixT = [0] * (n+1)
    for i in range(2, n+1):
        prefixT[i] = (prefixT[i-1] + T_arr[i]) % MOD
        
    res = []
    for _ in range(q):
        u = int(next(it))
        v = int(next(it))
        m = u
        
        s_u = 0
        if u >= 2:
            s_u = (prefix_arr1[u-1] + B[u]) % MOD
            
        s_v = 0
        if v >= 2:
            s_v = (prefix_arr1[v-1] + B[v]) % MOD
            
        total_s = (s_u + s_v) % MOD
        part1 = fac[n-1] * total_s % MOD
        
        part2 = 0
        if m >= 2:
            part2 = 2 * prefixT[m] % MOD
            
        ans = (part1 - part2) % MOD
        if ans < 0:
            ans += MOD
        res.append(str(ans))
        
    print("
".join(res))

if __name__ == "__main__":
    main()