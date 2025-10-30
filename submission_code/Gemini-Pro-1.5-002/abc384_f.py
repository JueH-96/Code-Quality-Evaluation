def f(x):
    while x % 2 == 0:
        x //= 2
    return x

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    ans = 0
    for i in range(n):
        for j in range(i, n):
            ans += f(a[i] + a[j])
    
    print(ans)

solve()