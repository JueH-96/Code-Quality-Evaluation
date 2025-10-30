# Read the input strings from stdin
strings = [input() for _ in range(12)]

# Count the number of strings whose length matches their index
count = 0
for i, s in enumerate(strings):
    if len(s) == i+1:
        count += 1

# Print the result
print(count)