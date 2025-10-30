K = int(input())

numbers = []
for mask in range(1, 2**10):
    digits = []
    for i in range(10):
        if mask & (1 << i):
            digits.append(i)
    digits.sort(reverse=True)
    if digits[0] > 0:
        number = int(''.join(map(str, digits)))
        numbers.append(number)

numbers.sort()
print(numbers[K-1])