# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    ans = 0
    mod = 998244353
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            ans = (ans + int(str(a[i]) + str(a[j]))) % mod
            
    print(ans)

solve()