s = input()
counts = {}
for char in s:
    if char in counts:
        counts[char] += 1
    else:
        counts[char] = 1

max_count = 0
most_frequent_char = ''
for char in sorted(counts.keys()):
    if counts[char] > max_count:
        max_count = counts[char]
        most_frequent_char = char
    elif counts[char] == max_count and (most_frequent_char == '' or char < most_frequent_char):
        most_frequent_char = char

print(most_frequent_char)