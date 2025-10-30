import sys
import math
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Precompute divisors for each number in A
    max_value = max(A)
    divisor_count = defaultdict(int)
    
    # Count how many numbers are divisible by each possible divisor
    for number in A:
        for d in range(1, int(math.sqrt(number)) + 1):
            if number % d == 0:
                divisor_count[d] += 1
                if d != number // d:
                    divisor_count[number // d] += 1
    
    # For each A_i, find the maximum GCD
    results = []
    for i in range(N):
        max_gcd = 1
        for d in range(1, int(math.sqrt(A[i])) + 1):
            if A[i] % d == 0:
                # Check if d can be the GCD
                if divisor_count[d] >= K:
                    max_gcd = max(max_gcd, d)
                # Check if A[i] // d can be the GCD
                if divisor_count[A[i] // d] >= K:
                    max_gcd = max(max_gcd, A[i] // d)
        results.append(max_gcd)
    
    # Output the results
    for result in results:
        print(result)