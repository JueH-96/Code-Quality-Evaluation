import sys
import math
from collections import Counter

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    # Count occurrences of each number
    counter = Counter(A)
    numbers = sorted(counter.keys())
    total_sum = 0

    # Precompute prefix sums for each unique number
    prefix_sums = {}
    current_sum = 0
    for num in numbers:
        prefix_sums[num] = current_sum
        current_sum += counter[num]

    # Calculate the sum based on the given formula
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            max_val = numbers[j]
            min_val = numbers[i]
            ratio = math.floor(max_val / min_val)
            count_pairs = counter[numbers[i]] * (prefix_sums[numbers[j]] - prefix_sums[numbers[i]])
            total_sum += ratio * count_pairs

    print(total_sum)

if __name__ == "__main__":
    solve()