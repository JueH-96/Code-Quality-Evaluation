import math
from collections import defaultdict

def get_divisors(n):
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Precompute for each possible divisor, which elements are divisible by it
    max_val = max(A)
    divisible_count = [0] * (max_val + 1)
    
    # Count how many elements are divisible by each number
    for num in A:
        for i in range(1, int(math.sqrt(num)) + 1):
            if num % i == 0:
                divisible_count[i] += 1
                if i != num // i:
                    divisible_count[num // i] += 1
    
    results = []
    
    for i in range(N):
        ai = A[i]
        max_gcd = 1
        
        # Get all divisors of A[i]
        divisors = get_divisors(ai)
        
        # For each divisor, check if we can form a set of K elements
        for d in divisors:
            if divisible_count[d] >= K:
                max_gcd = max(max_gcd, d)
        
        results.append(max_gcd)
    
    for result in results:
        print(result)

solve()