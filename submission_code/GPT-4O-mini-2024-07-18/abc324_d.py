from itertools import permutations

def count_square_numbers(N, S):
    square_numbers = set()
    max_possible_value = int('9' * N)  # The maximum number that can be formed with N digits of '9'
    max_square_root = int(max_possible_value**0.5) + 1
    squares = {i * i for i in range(max_square_root)}  # Precompute all square numbers up to the maximum possible value

    unique_numbers = set()
    for perm in permutations(S):
        num = int(''.join(perm))
        if num in squares:
            unique_numbers.add(num)

    return len(unique_numbers)

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Get the result and print it
result = count_square_numbers(N, S)
print(result)