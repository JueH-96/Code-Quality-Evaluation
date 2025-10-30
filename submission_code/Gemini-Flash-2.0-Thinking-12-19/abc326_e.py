def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def inverse(n, m):
    return power(n, m - 2, m)

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    inv_n = inverse(n, mod)
    r = ((n + 1) * inv_n) % mod
    current_r_power = 1
    total_sum = 0
    
    for i in range(n):
        term = (a[i] * current_r_power) % mod
        total_sum = (total_sum + term) % mod
        current_r_power = (current_r_power * r) % mod
        
    result = (total_sum * inv_n) % mod
    print(result)

if __name__ == '__main__':
    solve()