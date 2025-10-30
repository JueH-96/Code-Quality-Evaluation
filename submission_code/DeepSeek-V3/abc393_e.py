import sys
import math

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    N, K = map(int, input[ptr:ptr+2])
    ptr +=2
    A = list(map(int, input[ptr:ptr+N]))
    
    max_A = max(A)
    
    # Initialize frequency and count_multiples arrays
    freq = [0] * (max_A + 2)
    for num in A:
        freq[num] += 1
    
    # Compute count_multiples[d] = number of elements in A divisible by d
    count_multiples = [0] * (max_A + 2)
    for d in range(1, max_A + 1):
        for multiple in range(d, max_A + 1, d):
            count_multiples[d] += freq[multiple]
    
    # Precompute for each A_i the maximum d dividing A_i with count_multiples[d] >= K
    res = [0] * N
    for i in range(N):
        x = A[i]
        divisors = set()
        # Find all divisors of x
        for d in range(1, int(math.isqrt(x)) + 1):
            if x % d == 0:
                divisors.add(d)
                divisors.add(x // d)
        divisors = sorted(divisors, reverse=True)
        for d in divisors:
            if count_multiples[d] >= K:
                res[i] = d
                break
    print('
'.join(map(str, res)))

solve()