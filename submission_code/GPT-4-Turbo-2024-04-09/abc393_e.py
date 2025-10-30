import sys
import math
from collections import defaultdict

input = sys.stdin.read
def main():
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    max_value = max(A)
    
    # Frequency of each value in A
    freq = [0] * (max_value + 1)
    for num in A:
        freq[num] += 1
    
    # Result for each position
    results = []
    
    # To find the maximum GCD when including A_i, we need to consider all possible GCDs that A_i can contribute to.
    # We will iterate over all possible GCDs g that A_i could be a part of (i.e., divisors of A_i)
    # and then check how many numbers in A are multiples of g.
    for i in range(N):
        ai = A[i]
        max_gcd = 0
        
        # We will check all multiples of each divisor of ai
        # To find divisors efficiently, we iterate up to sqrt(ai) and consider pairs (d, ai//d)
        for d in range(1, int(math.sqrt(ai)) + 1):
            if ai % d == 0:
                # d is a divisor
                # Check multiples of d
                count = 0
                for multiple in range(d, max_value + 1, d):
                    count += freq[multiple]
                if count >= K:
                    max_gcd = max(max_gcd, d)
                
                # ai // d is another divisor if it's different from d
                if d != ai // d:
                    other_divisor = ai // d
                    count = 0
                    for multiple in range(other_divisor, max_value + 1, other_divisor):
                        count += freq[multiple]
                    if count >= K:
                        max_gcd = max(max_gcd, other_divisor)
        
        results.append(max_gcd)
    
    # Output results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()