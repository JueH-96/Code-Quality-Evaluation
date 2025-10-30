def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    expected_salary = 0
    
    for i in range(n):
        expected_salary = (expected_salary + a[i] * pow(n - i, mod - 2, mod)) % mod
        
    print(expected_salary)

solve()