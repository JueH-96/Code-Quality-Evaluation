K = int(input())

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
numbers = []

for mask in range(1, 1 << 10):
    if mask == 1:
        continue  # Skip the subset {0}
    subset = []
    for i in range(10):
        if mask & (1 << i):
            subset.append(digits[i])
    subset.sort(reverse=True)
    num = int(''.join(map(str, subset)))
    numbers.append(num)

numbers.sort()
print(numbers[K-1])