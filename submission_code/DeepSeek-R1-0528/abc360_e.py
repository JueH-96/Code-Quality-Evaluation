mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    
    if N == 1:
        print(1)
        return
        
    invN = pow(N, mod-2, mod)
    base = (N - 2) * invN % mod
    
    term = pow(base, K, mod)
    
    numerator = (N + 1) % mod
    temp = (N - 1) % mod
    temp = (temp * term) % mod
    numerator = (numerator - temp) % mod
    if numerator < 0:
        numerator += mod
        
    inv2 = pow(2, mod-2, mod)
    ans = numerator * inv2 % mod
    print(ans)

if __name__ == '__main__':
    main()