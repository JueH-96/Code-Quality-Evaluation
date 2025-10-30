def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    mod = 998244353
    ans = 0
    
    for i in range(1 << (n - 1)):
        valid = True
        current_subsequence = []
        for j in range(n):
            current_subsequence.append(a[j])
            if j < n - 1 and (i >> j) & 1:
                if sum(current_subsequence) == k:
                    valid = False
                    break
                current_subsequence = []
        if valid and sum(current_subsequence) != k:
            ans = (ans + 1) % mod
    
    print(ans)

solve()