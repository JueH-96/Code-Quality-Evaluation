from collections import Counter
s = input()
counter = Counter(s)
num_of_counts = Counter(counter.values())

is_good = True
for i, num in num_of_counts.items():
    if num != 0 and num != 2:
        is_good = False
        break
print('Yes' if is_good else 'No')