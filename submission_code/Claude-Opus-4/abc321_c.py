# YOUR CODE HERE
K = int(input())

# Generate all 321-like numbers
numbers = []

# For each possible subset of digits 1-9 (represented by bitmask)
for mask in range(1, 1024):  # 2^10 - 1 = 1023, but we start from 1
    # Build the number by including digits in decreasing order
    num = 0
    for digit in range(9, -1, -1):
        if mask & (1 << digit):
            num = num * 10 + digit
    numbers.append(num)

# Sort the numbers
numbers.sort()

# Output the K-th number (1-indexed)
print(numbers[K-1])