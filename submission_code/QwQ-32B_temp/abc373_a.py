# Read the 12 input strings
strings = [input().strip() for _ in range(12)]

count = 0
for i in range(12):
    if len(strings[i]) == i + 1:
        count += 1

print(count)