import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

if C == 0:
    min_A = min(A)
    print(K * min_A)
else:
    period = M // gcd(C, M)
    
    if K <= period:
        total_sum = 0
        for k in range(K):
            r = (C * k) % M
            g_r = min((r + a) % M for a in A)
            total_sum += g_r
        print(total_sum)
    else:
        # Compute the sum for one full period
        period_sum = 0
        for k in range(period):
            r = (C * k) % M
            g_r = min((r + a) % M for a in A)
            period_sum += g_r
        
        full_periods = K // period
        remaining = K % period
        
        total_sum = full_periods * period_sum
        
        # Add the sum for the remaining terms
        for k in range(remaining):
            r = (C * k) % M
            g_r = min((r + a) % M for a in A)
            total_sum += g_r
        
        print(total_sum)