mod = 998244353

import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    P = [0] * (n + 1)
    for i in range(1, n + 1):
        P[i] = P[i - 1] + A[i - 1]
    
    freq = defaultdict(int)
    freq[0] = 1
    total = 1
    
    for i in range(1, n + 1):
        s = P[i]
        target = s - k
        cnt = freq.get(target, 0)
        dp_i = (total - cnt) % mod
        total = (total + dp_i) % mod
        freq[s] = (freq.get(s, 0) + dp_i) % mod
        
    print(dp_i % mod)

if __name__ == "__main__":
    main()