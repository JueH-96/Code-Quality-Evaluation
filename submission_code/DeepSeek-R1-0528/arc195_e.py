MOD = 998244353

import sys

def modinv(a, mod=MOD):
    return pow(a, mod-2, mod)

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    q = int(data[1])
    A = [0] * (n+1)
    index = 2
    for i in range(2, n+1):
        A[i] = int(data[index]); index += 1
    
    fact = [1] * (n+1)
    for i in range(1, n+1):
        fact[i] = fact[i-1] * i % MOD
    
    s = [0] * (n+1)
    for i in range(2, n+1):
        inv_i = modinv(i)
        term = A[i] * inv_i % MOD
        s[i] = (s[i-1] + term) % MOD
    
    ans_lines = []
    for _ in range(q):
        u = int(data[index]); v = int(data[index+1]); index += 2
        if u > v:
            u, v = v, u
        total1 = s[u] * fact[n-1] % MOD
        total2 = s[v] * fact[n-1] % MOD
        total3 = s[u] * fact[n-1] % MOD
        total = (total1 + total2 - 2 * total3) % MOD
        ans_lines.append(str(total))
    
    print("
".join(ans_lines))

if __name__ == '__main__':
    main()