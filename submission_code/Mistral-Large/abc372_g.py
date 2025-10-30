import sys
import math
from collections import deque

def count_valid_pairs(N, ABC):
    # Initialize the constraints
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    for A, B, C in ABC:
        # Calculate the maximum possible x and y values
        max_x = min(max_x, (C - 1) // A)
        max_y = min(max_y, (C - 1) // B)

        # Calculate the minimum possible x and y values
        if A * (C - 1) // B + 1 < C:
            min_x = min(min_x, (C - 1) // (A * (C - 1) // B + 1))
        if B * (C - 1) // A + 1 < C:
            min_y = min(min_y, (C - 1) // (B * (C - 1) // A + 1))

    # Ensure the minimum values are at least 1
    min_x = max(1, min_x)
    min_y = max(1, min_y)

    # Ensure the maximum values are at least the minimum values
    max_x = max(min_x, max_x)
    max_y = max(min_y, max_y)

    # Calculate the number of valid pairs
    if max_x >= min_x and max_y >= min_y:
        return max(0, (max_x - min_x + 1) * (max_y - min_y + 1))
    else:
        return 0

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        N = int(data[index])
        index += 1
        ABC = []
        for __ in range(N):
            A = int(data[index])
            B = int(data[index + 1])
            C = int(data[index + 2])
            index += 3
            ABC.append((A, B, C))
        results.append(count_valid_pairs(N, ABC))

    for result in results:
        print(result)

if __name__ == "__main__":
    main()