from itertools import combinations

def generate_321_like_numbers():
    digits = list(range(10))
    numbers = set()
    for r in range(1, 10):
        for combo in combinations(digits, r):
            combo = sorted(combo, reverse=True)
            number = int(''.join(map(str, combo)))
            numbers.add(number)
    return sorted(numbers)

def find_kth_321_like_number(k):
    numbers = generate_321_like_numbers()
    return numbers[k-1]

# Read input
import sys
input = sys.stdin.read
K = int(input().strip())

# Find and print the K-th smallest 321-like number
result = find_kth_321_like_number(K)
print(result)