from collections import Counter

s = input().strip()
char_counts = Counter(s)
freq_counts = Counter(char_counts.values())

is_good = all(count == 2 for count in freq_counts.values())
print("Yes" if is_good else "No")