from itertools import combinations

K = int(input())

numbers = []

# Generate all 321-like numbers
for length in range(1, 11):  # Consider subsets of length 1 to 10
    for combo in combinations(range(10), length):
        # Skip the subset {0} as it would give 0, not a positive integer
        if combo == (0,):
            continue
        
        # Sort digits in decreasing order
        digits = sorted(combo, reverse=True)
        
        # Convert to number
        num = 0
        for d in digits:
            num = num * 10 + d
        
        numbers.append(num)

# Sort all numbers
numbers.sort()

# Output the K-th smallest (1-indexed)
print(numbers[K-1])