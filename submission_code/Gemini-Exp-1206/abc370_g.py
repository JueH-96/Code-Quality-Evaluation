def solve():
    n, m = map(int, input().split())
    mod = 998244353

    def divisors(num):
        divs = []
        i = 1
        while i * i <= num:
            if num % i == 0:
                divs.append(i)
                if i * i != num:
                    divs.append(num // i)
            i += 1
        return divs

    def sum_divisors(num):
        return sum(divisors(num))

    def is_good(num):
        return sum_divisors(num) % 3 == 0

    good_nums = []
    for i in range(1, min(n + 1, 1001)):
        if is_good(i):
            good_nums.append(i)
            
    
    dp = {}

    def count_sequences(current_product, length):
        if length == m:
            return 1 if is_good(current_product) and current_product <= n else 0

        if (current_product, length) in dp:
            return dp[(current_product, length)]

        count = 0
        
        
        if current_product > n:
            dp[(current_product, length)] = 0
            return 0

        
        limit = n // current_product
        
        
        for i in range(1, limit + 1):
            count = (count + count_sequences(current_product * i, length + 1)) % mod

        dp[(current_product, length)] = count
        return count

    ans = count_sequences(1, 0)
    print(ans)

solve()