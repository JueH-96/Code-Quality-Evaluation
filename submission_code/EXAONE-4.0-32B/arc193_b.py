mod = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    k = s.count('1')
    
    power2n = pow(2, n, mod)
    distinct_base = (power2n - 1) % mod
    power2k = pow(2, k, mod)
    
    ans = (distinct_base * power2k) % mod
    print(ans)

if __name__ == "__main__":
    main()