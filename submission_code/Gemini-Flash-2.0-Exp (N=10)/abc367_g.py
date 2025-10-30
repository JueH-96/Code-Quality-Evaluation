def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    mod = 998244353
    total_score = 0
    
    for i in range(1, 1 << n):
        subsequence = []
        for j in range(n):
            if (i >> j) & 1:
                subsequence.append(a[j])
        
        subsequence_len = len(subsequence)
        if subsequence_len == 0:
            continue
            
        if subsequence_len % m == 0:
            xor_sum = 0
            for x in subsequence:
                xor_sum ^= x
            
            score = pow(xor_sum, k, mod)
            total_score = (total_score + score) % mod
            
    print(total_score)

solve()