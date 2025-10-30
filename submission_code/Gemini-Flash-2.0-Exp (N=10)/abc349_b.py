from collections import Counter

s = input()
counts = Counter(s)
freq_counts = Counter(counts.values())

for freq, count in freq_counts.items():
    if count != 0 and count != 2:
        print("No")
        exit()
print("Yes")