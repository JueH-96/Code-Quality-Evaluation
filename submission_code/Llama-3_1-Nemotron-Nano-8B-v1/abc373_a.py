# Read the 12 strings from input
strings = [input() for _ in range(12)]

# Count how many strings have length equal to their 1-based index
count = 0
for i in range(12):
    if len(strings[i]) == i + 1:
        count += 1

# Output the result
print(count)