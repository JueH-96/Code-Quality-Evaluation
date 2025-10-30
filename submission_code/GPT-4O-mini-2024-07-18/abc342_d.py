def count_square_pairs(N, A):
    from collections import defaultdict
    from math import isqrt

    # Count occurrences of each number
    count = defaultdict(int)
    for number in A:
        count[number] += 1

    # To store the number of valid pairs
    total_pairs = 0

    # Check pairs of numbers
    for a in count:
        for b in count:
            if a <= b:  # Ensure we only consider pairs (a, b) where a <= b
                product = a * b
                if product == 0:  # (0, 0) is a square
                    total_pairs += count[a] * (count[a] - 1) // 2  # Choose 2 from count[a]
                elif isqrt(product) ** 2 == product:  # Check if product is a perfect square
                    if a == b:
                        total_pairs += count[a] * (count[a] - 1) // 2  # Choose 2 from count[a]
                    else:
                        total_pairs += count[a] * count[b]  # All combinations of count[a] and count[b]

    return total_pairs

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))

result = count_square_pairs(N, A)
print(result)