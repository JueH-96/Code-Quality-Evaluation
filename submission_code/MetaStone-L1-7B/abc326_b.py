import bisect

n = int(input())

# Generate all 326-like numbers
numbers = []
for a in range(1, 10):  # Hundreds digit (1-9)
    for b in range(0, 10):  # Tens digit (0-9)
        c = a * b
        if c < 10:  # Ones digit must be a single digit
            num = a * 100 + b * 10 + c
            numbers.append(num)

# Sort the list of numbers
numbers.sort()

# Find the smallest number >= n
index = bisect.bisect_left(numbers, n)
print(numbers[index])