# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    mod = 998244353

    def sum_divisors(x):
        res = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                res += i
                if i * i != x:
                    res += x // i
            i += 1
        return res

    good_integers = []
    for i in range(1, int(n**0.5) + 2):
        if i * i > n:
            break
        if sum_divisors(i) % 3 == 0:
            good_integers.append(i)
        if i * i != n and sum_divisors(n // i) % 3 == 0:
            good_integers.append(n // i)
    
    good_integers.sort()
    
    good_count = 0
    for x in good_integers:
        if x <= n:
            good_count += 1

    
    ans = 0
    
    def power(a, b):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res

    total_count = power(n, m)
    
    
    ans = 0
    for x in good_integers:
        if x <= n:
            ans = (ans + power(x,m)) % mod
            
    print(ans)
    

solve()