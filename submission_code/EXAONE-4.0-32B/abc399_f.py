mod = 998244353
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    binom = [0] * (k+1)
    binom[0] = 1
    for j in range(1, k+1):
        binom[j] = binom[j-1] * (k - j + 1) // j
        
    coef = [0] * (k+1)
    for j in range(k+1):
        if j % 2 == 0:
            coef[j] = binom[j] % mod
        else:
            coef[j] = (-binom[j]) % mod
    
    prefix = 0
    S = [0] * (k+1)
    S[0] = 1
    
    ans = 0
    for i in range(n):
        prefix = (prefix + A[i]) % mod
        
        power_arr = [1] * (k+1)
        if k >= 1:
            power_arr[1] = prefix
            for j in range(2, k+1):
                power_arr[j] = power_arr[j-1] * prefix % mod
                
        for j in range(k+1):
            exp = k - j
            term = power_arr[exp] * S[j] % mod
            term = term * coef[j] % mod
            ans = (ans + term) % mod
            
        for j in range(k+1):
            S[j] = (S[j] + power_arr[j]) % mod
            
    print(ans % mod)

if __name__ == "__main__":
    main()