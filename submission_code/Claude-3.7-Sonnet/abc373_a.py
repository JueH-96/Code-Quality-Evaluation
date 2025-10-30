# Read 12 strings
strings = [input().strip() for _ in range(12)]

# Count how many strings have length equal to their position
count = 0
for i in range(1, 13):
    if len(strings[i-1]) == i:
        count += 1

# Print the result
print(count)