def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_score = 0
    mod = 998244353
    
    for i in range(1, 1 << n):
        subsequence = []
        for j in range(n):
            if (i >> j) & 1:
                subsequence.append(a[j])
        
        if not subsequence:
            continue
        
        xor_sum = 0
        for val in subsequence:
            xor_sum ^= val
        
        if len(subsequence) % m == 0:
            score = pow(xor_sum, k, mod)
            total_score = (total_score + score) % mod
    
    print(total_score)

solve()