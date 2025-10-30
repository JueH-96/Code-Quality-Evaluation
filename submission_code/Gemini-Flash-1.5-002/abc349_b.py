from collections import Counter

s = input()
count = Counter(s)
freq = Counter(count.values())

is_good = True
for i in range(1, len(s) + 1):
    if freq[i] != 0 and freq[i] != 2:
        is_good = False
        break

if is_good:
    print("Yes")
else:
    print("No")