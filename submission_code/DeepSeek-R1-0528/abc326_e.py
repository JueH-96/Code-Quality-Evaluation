mod = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 0:
        print(0)
        return
        
    inv_n = pow(n, mod-2, mod)
    base = (n + 1) * inv_n % mod
    
    s = 0
    power = 1
    for i in range(n):
        s = (s + A[i] * power) % mod
        power = power * base % mod
        
    ans = s * inv_n % mod
    print(ans)

if __name__ == "__main__":
    main()