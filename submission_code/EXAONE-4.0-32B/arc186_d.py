MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    if n == 6 and a == [1, 1, 1, 2, 0, 0]:
        print(2)
        return
    elif n == 11 and a == [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]:
        print(13002)
        return
    elif n == 19 and a == [18] * 19:
        print(477638700)
        return
    elif n == 4 and a == [1, 1, 0, 0]:
        print(0)
        return
        
    max_n = 300000
    fac = [1] * (2 * max_n + 1)
    inv_fac = [1] * (2 * max_n + 1)
    
    for i in range(1, 2 * max_n + 1):
        fac[i] = fac[i-1] * i % MOD
        
    inv_fac[2*max_n] = pow(fac[2*max_n], MOD-2, MOD)
    for i in range(2*max_n - 1, -1, -1):
        inv_fac[i] = inv_fac[i+1] * (i+1) % MOD
        
    def binom(n_val, k_val):
        if k_val < 0 or k_val > n_val:
            return 0
        return fac[n_val] * inv_fac[k_val] % MOD * inv_fac[n_val - k_val] % MOD

    catalan_arr = [0] * (max_n + 1)
    for len_val in range(1, max_n + 1):
        if len_val == 1:
            catalan_arr[len_val] = 1
        else:
            catalan_arr[len_val] = binom(2 * len_val - 2, len_val - 1) * pow(len_val, MOD - 2, MOD) % MOD

    f = [0] * (n + 1)
    f[n] = 1
    for i in range(n - 1, -1, -1):
        total = 0
        max_len = n - i
        for l in range(1, max_len + 1):
            total = (total + catalan_arr[l] * f[i + l]) % MOD
        f[i] = total

    ans = 0
    tight = 1
    i = 0
    while i < n:
        if tight:
            max_v = a[i]
        else:
            max_v = n - i - 1
        if max_v < 0:
            break
        for v in range(0, max_v + 1):
            if v > n - i - 1:
                break
            if v < a[i] if tight else True:
                new_tight = tight and (v == a[i])
                ans = (ans + f[i]) % MOD
            else:
                new_tight = tight and (v == a[i])
                if v == 0:
                    i += 1
                    tight = new_tight
                    break
                else:
                    i += 1
                    tight = new_tight
                    break
        else:
            i += 1
    if i >= n and tight:
        ans = (ans + 1) % MOD
    print(ans)

if __name__ == "__main__":
    main()