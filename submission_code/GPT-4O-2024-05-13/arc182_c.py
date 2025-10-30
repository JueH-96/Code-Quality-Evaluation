# YOUR CODE HERE
import sys
import math

MOD = 998244353

def count_divisors(x):
    count = 0
    for i in range(1, int(math.sqrt(x)) + 1):
        if x % i == 0:
            count += 1
            if i != x // i:
                count += 1
    return count

def solve(N, M):
    # Precompute the number of divisors for all products up to M^N
    max_product = M ** N
    divisor_counts = [0] * (max_product + 1)
    
    for i in range(1, max_product + 1):
        divisor_counts[i] = count_divisors(i)
    
    total_score = 0
    
    # Sum the scores for all sequences of length 1 to N
    for length in range(1, N + 1):
        for sequence in range(1, M ** length + 1):
            total_score += divisor_counts[sequence]
            total_score %= MOD
    
    print(total_score)

if __name__ == "__main__":
    input = sys.stdin.read().strip()
    N, M = map(int, input.split())
    solve(N, M)