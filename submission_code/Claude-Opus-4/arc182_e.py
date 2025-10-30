import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

# Calculate the period
g = gcd(C, M)
period = M // g

# If K is small enough, just compute directly
if K <= period:
    total = 0
    for k in range(K):
        min_val = M
        for a in A:
            val = (C * k + a) % M
            min_val = min(min_val, val)
        total += min_val
    print(total)
else:
    # Compute sum for one complete cycle
    cycle_sum = 0
    for k in range(period):
        min_val = M
        for a in A:
            val = (C * k + a) % M
            min_val = min(min_val, val)
        cycle_sum += min_val
    
    # Number of complete cycles
    complete_cycles = K // period
    remaining = K % period
    
    # Total sum
    total = cycle_sum * complete_cycles
    
    # Add remaining partial cycle
    for k in range(remaining):
        min_val = M
        for a in A:
            val = (C * k + a) % M
            min_val = min(min_val, val)
        total += min_val
    
    print(total)