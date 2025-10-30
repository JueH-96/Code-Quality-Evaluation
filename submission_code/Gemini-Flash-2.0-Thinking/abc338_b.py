from collections import Counter

s = input()
counts = Counter(s)
max_freq = 0
for count in counts.values():
    max_freq = max(max_freq, count)

most_frequent_chars = []
for char, count in counts.items():
    if count == max_freq:
        most_frequent_chars.append(char)

most_frequent_chars.sort()
print(most_frequent_chars[0])