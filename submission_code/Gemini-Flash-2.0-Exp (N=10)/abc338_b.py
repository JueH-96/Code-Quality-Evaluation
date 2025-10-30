s = input()
counts = {}
for char in s:
    if char in counts:
        counts[char] += 1
    else:
        counts[char] = 1

max_count = 0
result = ''
for char, count in counts.items():
    if count > max_count:
        max_count = count
        result = char
    elif count == max_count and char < result:
        result = char
print(result)