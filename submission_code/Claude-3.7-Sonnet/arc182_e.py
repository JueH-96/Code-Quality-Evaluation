# YOUR CODE HERE
import math

def solve():
    N, M, C, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Find the period of (C*k) % M
    g = math.gcd(C, M)
    period = M // g
    
    # Compute the minimum for each k in the period
    X_values = []  # Stores min_{i} { (C*k + A_i) % M } for each k in the period
    
    for k in range(period):
        r_k = (C * k) % M  # Compute (C*k) % M
        
        # Compute min_{i} { (C*k + A_i) % M }
        min_val = float('inf')
        for i in range(N):
            remainder = (r_k + A[i]) % M
            min_val = min(min_val, remainder)
        
        X_values.append(min_val)
    
    # Compute the sum for all k from 0 to K-1
    total_sum = 0
    
    # Sum over complete periods
    complete_periods = K // period
    total_sum += complete_periods * sum(X_values)
    
    # Sum over the remainder
    remaining_k = K % period
    total_sum += sum(X_values[:remaining_k])
    
    return total_sum

print(solve())