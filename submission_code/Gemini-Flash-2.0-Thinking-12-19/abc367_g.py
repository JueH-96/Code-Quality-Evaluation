def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    mod = 998244353
    
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    def inverse(n):
        return power(n, mod - 2)
        
    def xor_sum(subsequence):
        res = 0
        for x in subsequence:
            res ^= x
        return res
        
    total_score = 0
    for i in range(1, 1 << n):
        subsequence = []
        for j in range(n):
            if (i >> j) & 1:
                subsequence.append(a[j])
        length = len(subsequence)
        if length == 0:
            continue
        if length % m == 0:
            current_xor = xor_sum(subsequence)
            score = power(current_xor, k)
            total_score = (total_score + score) % mod
            
    print(total_score)

if __name__ == '__main__':
    solve()