from collections import Counter

s = input().strip()
char_count = Counter(s)
count_freq = Counter(char_count.values())

for v in count_freq.values():
    if v != 2:
        print("No")
        exit()
print("Yes")