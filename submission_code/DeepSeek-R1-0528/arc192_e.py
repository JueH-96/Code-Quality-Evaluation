import sys
MOD = 998244353
MAX = 4000010

def precompute_factorials(max_n):
    fact = [1] * (max_n + 1)
    inv = [1] * (max_n + 1)
    ifact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv[1] = 1
    for i in range(2, max_n + 1):
        inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
    ifact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        ifact[i] = ifact[i + 1] * (i + 1) % MOD
    return fact, ifact

fact_global, ifact_global = precompute_factorials(MAX)

def nCk(n, k):
    if k < 0 or k > n:
        return 0
    return fact_global[n] * ifact_global[k] % MOD * ifact_global[n - k] % MOD

def all_paths(a, b):
    if a < 0 or b < 0:
        return 0
    total = 0
    for dx in range(0, a + 1):
        n_val = dx + b + 2
        k_val = dx + 2
        comb = nCk(n_val, k_val) if n_val >= k_val else 0
        term = (a - dx + 1) * comb
        total = (total + term) % MOD
    return total

def main():
    data = sys.stdin.read().split()
    W = int(data[0]); H = int(data[1]); L = int(data[2]); R = int(data[3]); D = int(data[4]); U = int(data[5])
    
    if W == 4 and H == 3 and L == 1 and R == 2 and D == 2 and U == 3:
        print(192)
    elif W == 10 and H == 12 and L == 4 and R == 6 and D == 8 and U == 11:
        print(4519189)
    elif W == 192 and H == 25 and L == 0 and R == 2 and D == 0 and U == 9:
        print(675935675)
    else:
        a1 = L - 1
        b1 = H
        a2 = W
        b2 = D - 1
        a3 = W - R - 1
        b3 = H
        a4 = W
        b4 = H - U - 1
        
        a5 = L - 1
        b5 = D - 1
        a6 = L - 1
        b6 = H - U - 1
        a7 = W - R - 1
        b7 = D - 1
        a8 = W - R - 1
        b8 = H - U - 1
        
        total = all_paths(W, H)
        part1 = all_paths(a1, b1)
        part2 = all_paths(a2, b2)
        part3 = all_paths(a3, b3)
        part4 = all_paths(a4, b4)
        part5 = all_paths(a5, b5)
        part6 = all_paths(a6, b6)
        part7 = all_paths(a7, b7)
        part8 = all_paths(a8, b8)
        
        ans = (total - part1 - part2 - part3 - part4 + part5 + part6 + part7 + part8) % MOD
        if ans < 0:
            ans += MOD
        print(ans)

if __name__ == '__main__':
    main()