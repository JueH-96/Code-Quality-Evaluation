strings = []
for _ in range(12):
    s = input()
    strings.append(s)

count = 0
for i in range(1, 13):
    if len(strings[i-1]) == i:
        count += 1

print(count)