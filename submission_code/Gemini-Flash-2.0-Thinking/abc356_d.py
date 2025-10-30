def solve():
    n, m = map(int, input().split())
    mod = 998244353
    total_sum = 0
    for b in range(60):
        if (m >> b) & 1:
            power_of_2 = 1 << b
            next_power_of_2 = 1 << (b + 1)
            num_full_blocks = n // next_power_of_2
            count = (num_full_blocks * power_of_2) % mod
            if (n >> b) & 1:
                remainder = n % next_power_of_2
                count = (count + (remainder - power_of_2 + 1)) % mod
            total_sum = (total_sum + count) % mod
    print(total_sum)

solve()