# YOUR CODE HERE
def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    ans = 0
    for i in range(1, 1 << n):
        subsequence = []
        for j in range(n):
            if (i >> j) & 1:
                subsequence.append(a[j])
        
        l = len(subsequence)
        if l % m == 0:
            xor_sum = 0
            for x in subsequence:
                xor_sum ^= x
            
            score = pow(xor_sum, k, mod)
            ans = (ans + score) % mod
    
    print(ans)

solve()