def solve():
    n, m = map(int, input().split())
    total_sum = 0
    mod = 998244353
    for i in range(60):
        m_i = (m >> i) & 1
        if m_i == 1:
            prefix = n >> (i + 1)
            current_bit = (n >> i) & 1
            suffix = n & ((1 << i) - 1)
            count_i = (prefix * (1 << i)) + (current_bit * (suffix + 1))
            total_sum += count_i
            
    print(total_sum % mod)

if __name__ == '__main__':
    solve()