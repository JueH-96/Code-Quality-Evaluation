n = int(input())
s = input().strip()

from collections import defaultdict

max_runs = defaultdict(int)

if n == 0:
    print(0)
    exit()

current_char = s[0]
current_length = 1

for i in range(1, n):
    if s[i] == current_char:
        current_length += 1
    else:
        if current_length > max_runs[current_char]:
            max_runs[current_char] = current_length
        current_char = s[i]
        current_length = 1

# Update the last run
if current_length > max_runs[current_char]:
    max_runs[current_char] = current_length

total = sum(max_runs.values())

print(total)