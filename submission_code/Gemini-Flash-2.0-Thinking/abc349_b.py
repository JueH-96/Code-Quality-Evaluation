from collections import Counter

s = input()
frequencies = Counter(s)
frequency_counts = list(frequencies.values())
max_freq = max(frequency_counts) if frequency_counts else 0

for i in range(1, max_freq + 1):
    count_i = frequency_counts.count(i)
    if count_i != 0 and count_i != 2:
        print("No")
        exit()

print("Yes")