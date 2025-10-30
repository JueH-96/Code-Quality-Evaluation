count = 0
for i in range(12):
    s = input().strip()
    if len(s) == i + 1:
        count += 1
print(count)