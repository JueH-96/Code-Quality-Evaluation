n = int(input())
s = input().strip()

if n == 0:
    print(0)
    exit()

runs = []
current_char = s[0]
count = 1

for c in s[1:]:
    if c == current_char:
        count += 1
    else:
        runs.append((current_char, count))
        current_char = c
        count = 1
runs.append((current_char, count))

from collections import defaultdict

max_runs = defaultdict(int)

for char, length in runs:
    if length > max_runs[char]:
        max_runs[char] = length

total = sum(max_runs.values())
print(total)