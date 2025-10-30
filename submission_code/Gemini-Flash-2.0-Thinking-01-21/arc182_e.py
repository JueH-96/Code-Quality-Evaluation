import math

def solve():
    n, m, c, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    if c == 0:
        min_a = min(a)
        print(min_a * k)
        return
        
    g = math.gcd(c, m)
    period = m // g
    
    period_sum = 0
    m_values_period = []
    for current_k in range(period):
        min_val = float('inf')
        for i in range(n):
            val = (c * current_k + a[i]) % m
            min_val = min(min_val, val)
        period_sum += min_val
        m_values_period.append(min_val)
        
    num_periods = k // period
    remainder = k % period
    
    remainder_sum = 0
    for i in range(remainder):
        remainder_sum += m_values_period[i]
        
    total_sum = num_periods * period_sum + remainder_sum
    print(total_sum)

if __name__ == '__main__':
    solve()