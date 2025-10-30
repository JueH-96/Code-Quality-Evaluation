def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    def get_digit_count(num):
        return len(str(num))
        
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    digit_counts = []
    powers_of_10 = []
    for x in a:
        digits = get_digit_count(x)
        digit_counts.append(digits)
        powers_of_10.append(power(10, digits))
        
    prefix_sum = [0] * n
    prefix_sum[0] = a[0]
    for i in range(1, n):
        prefix_sum[i] = (prefix_sum[i-1] + a[i]) % mod
        
    total_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            val1 = a[i]
            val2 = a[j]
            digits_val2 = digit_counts[j]
            power_of_10_val2 = powers_of_10[j]
            f_val = (val1 * power_of_10_val2 + val2) % mod
            total_sum = (total_sum + f_val) % mod
            
    print(total_sum)

if __name__ == '__main__':
    solve()