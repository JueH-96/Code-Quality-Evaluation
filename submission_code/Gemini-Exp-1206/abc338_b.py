s = input()
counts = {}
for char in s:
    counts[char] = counts.get(char, 0) + 1

max_count = 0
result = ''
for char, count in counts.items():
    if count > max_count:
        max_count = count
        result = char
    elif count == max_count:
        result = min(result, char)

print(result)