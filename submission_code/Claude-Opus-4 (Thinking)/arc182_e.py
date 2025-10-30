# YOUR CODE HERE
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

N, M, C, K = map(int, input().split())
A = list(map(int, input().split()))

if C == 0:
    # All values are constant
    print(min(A) * K)
else:
    g = gcd(C, M)
    period = M // g
    
    if K <= period:
        # K is within one period, compute directly
        total = 0
        for k in range(K):
            min_val = min((C * k + a) % M for a in A)
            total += min_val
        print(total)
    else:
        # K spans multiple periods, use periodicity
        # Compute sum for one period
        period_sum = 0
        for k in range(period):
            min_val = min((C * k + a) % M for a in A)
            period_sum += min_val
        
        full_periods = K // period
        remaining = K % period
        
        # Sum for remaining steps
        remaining_sum = 0
        for k in range(remaining):
            min_val = min((C * k + a) % M for a in A)
            remaining_sum += min_val
        
        total = full_periods * period_sum + remaining_sum
        print(total)