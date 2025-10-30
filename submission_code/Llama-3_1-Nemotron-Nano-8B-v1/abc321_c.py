import itertools

k = int(input())

numbers = []

for d in range(1, 11):
    for comb in itertools.combinations(range(10), d):
        rev_comb = tuple(reversed(comb))
        if rev_comb[0] == 0:
            continue
        num = int(''.join(map(str, rev_comb)))
        numbers.append(num)

numbers.sort()

print(numbers[k-1])