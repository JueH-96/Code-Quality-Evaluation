def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    def get_digits(x):
        count = 0
        while x > 0:
            x //= 10
            count += 1
        return count
        
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    digits_count = [get_digits(x) for x in a]
    powers_of_10 = [power(10, d) for d in digits_count]
    suffix_sums = [0] * (n + 1)
    suffix_sums[n] = 0
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = (powers_of_10[i] + suffix_sums[i + 1]) % mod
        
    part1 = 0
    for i in range(n - 1):
        part1 = (part1 + a[i] * suffix_sums[i + 2]) % mod
        
    part2 = 0
    for j in range(1, n):
        part2 = (part2 + (j) * a[j]) % mod
        
    result = (part1 + part2) % mod
    print(result)

if __name__ == '__main__':
    solve()