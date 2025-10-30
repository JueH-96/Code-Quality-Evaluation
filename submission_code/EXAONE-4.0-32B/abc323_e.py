mod = 998244353
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    X = int(data[1])
    T_list = list(map(int, data[2:2+n]))
    T1 = T_list[0]
    
    dp = [0] * (X+1)
    dp[0] = 1
    inv_n = pow(n, mod-2, mod)
    
    for t in range(0, X+1):
        if dp[t] == 0:
            continue
        for d in T_list:
            nt = t + d
            if nt <= X:
                dp[nt] = (dp[nt] + dp[t] * inv_n) % mod
                
    ans = 0
    for s in range(0, X+1):
        if X < s + T1:
            ans = (ans + dp[s] * inv_n) % mod
            
    print(ans)

if __name__ == "__main__":
    main()