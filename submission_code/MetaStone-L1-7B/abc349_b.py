from collections import Counter

s = input().strip()
freq = Counter(s)
max_i = max(freq.values()) if freq else 0

for i in range(1, max_i + 1):
    count = 0
    for char, cnt in freq.items():
        if cnt == i:
            count += 1
    if count not in (0, 2):
        print("No")
        exit()
print("Yes")