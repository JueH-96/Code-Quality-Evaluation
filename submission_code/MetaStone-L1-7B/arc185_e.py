import sys
import math

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    max_a = 100000
    divisors = [[] for _ in range(max_a + 1)]
    for i in range(1, max_a + 1):
        for j in range(i, max_a + 1, i):
            divisors[j].append(i)
    for m in range(1, max_a + 1):
        divisors[m].sort(reverse=True)
    
    pow2 = [1] * (N + 1)
    for i in range(1, N + 1):
        pow2[i] = (pow2[i-1] * 2) % MOD
    
    f = {}
    ans = 0
    result = []
    
    for j in range(N):
        a = A[j]
        divs = divisors[a]
        current_sum = 0
        exact = {}
        for d in divs:
            e = (f.get(d, 0) - current_sum) % MOD
            exact[d] = e
            current_sum = (current_sum + e) % MOD
        
        S_j = 0
        for d in divs:
            term = (d * exact[d]) % MOD
            S_j = (S_j + term) % MOD
        
        ans = (ans * 2 + S_j) % MOD
        result.append(ans)
        
        for d in divs:
            val = pow2[j]
            f_val = f.get(d, 0)
            f[d] = (f_val + val) % MOD
    
    print('
'.join(map(str, result)))

if __name__ == '__main__':
    main()