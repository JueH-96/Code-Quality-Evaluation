import sys
import math
from itertools import accumulate

def max_rating(N, performances):
    max_rating = -float('inf')

    # Precompute powers of 0.9
    powers_of_09 = [0.9 ** i for i in range(N)]

    # Try all possible end positions for the contests
    for k in range(1, N + 1):
        # Calculate the sum of performances with decay
        decayed_sum = sum(powers_of_09[k - i] * performances[i - 1] for i in range(1, k + 1))
        # Calculate the sum of the decay factors
        decay_factor_sum = sum(powers_of_09[k - i] for i in range(1, k + 1))
        # Calculate the rating
        rating = (decayed_sum / decay_factor_sum) - (1200 / math.sqrt(k))
        # Update the max rating
        max_rating = max(max_rating, rating)

    return max_rating

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    performances = list(map(int, data[1:]))

    result = max_rating(N, performances)
    print(result)

if __name__ == "__main__":
    main()