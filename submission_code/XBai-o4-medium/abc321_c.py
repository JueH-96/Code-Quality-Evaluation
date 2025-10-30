import itertools

k = int(input())

numbers = []

for m in range(1, 11):
    for combo in itertools.combinations(range(10), m):
        if m == 1:
            if combo[0] == 0:
                continue
        digits = sorted(combo, reverse=True)
        num = 0
        for d in digits:
            num = num * 10 + d
        numbers.append(num)

numbers.sort()

print(numbers[k-1])