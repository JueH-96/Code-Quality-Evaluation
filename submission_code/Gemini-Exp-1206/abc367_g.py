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
        
        if len(subsequence) % m == 0:
            xor_sum = 0
            for num in subsequence:
                xor_sum ^= num
            ans = (ans + pow(xor_sum, k, mod)) % mod
    
    print(ans)

solve()