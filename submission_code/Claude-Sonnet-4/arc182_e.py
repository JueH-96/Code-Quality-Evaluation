# YOUR CODE HERE
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

if C == 0:
    # If C is 0, then for all k, we have min{A_i mod M} = min{A_i}
    min_val = min(A)
    print(min_val * K)
else:
    # Find the period of the sequence
    # The period is M / gcd(M, C)
    period = M // gcd(M, C)
    
    # If K is small or period is large, just compute directly
    if K <= period or period > 10**6:
        total = 0
        for k in range(K):
            min_val = float('inf')
            for a in A:
                val = (C * k + a) % M
                min_val = min(min_val, val)
            total += min_val
        print(total)
    else:
        # Compute one full period
        period_sum = 0
        period_mins = []
        
        for k in range(period):
            min_val = float('inf')
            for a in A:
                val = (C * k + a) % M
                min_val = min(min_val, val)
            period_mins.append(min_val)
            period_sum += min_val
        
        # How many complete periods fit in K?
        full_periods = K // period
        remaining = K % period
        
        total = full_periods * period_sum
        
        # Add the remaining partial period
        for k in range(remaining):
            total += period_mins[k]
        
        print(total)