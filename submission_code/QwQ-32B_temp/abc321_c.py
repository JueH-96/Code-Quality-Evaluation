K = int(input())

numbers = []
for mask in range(1, 2 ** 10):
    if mask == 1:  # Skip the subset {0}
        continue
    digits = []
    for i in range(10):
        if mask & (1 << i):
            digits.append(i)
    # Sort digits in descending order and form the number
    digits_sorted = sorted(digits, reverse=True)
    num = int(''.join(map(str, digits_sorted)))
    numbers.append(num)

# Sort all numbers in ascending order
numbers.sort()

print(numbers[K-1])