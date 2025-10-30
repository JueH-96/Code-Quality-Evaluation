# Read the 12 input strings
strings = [input().strip() for _ in range(12)]

# Initialize the count
count = 0

# Check each string's length against its 1-based index
for index, s in enumerate(strings):
    if len(s) == index + 1:
        count += 1

# Output the result
print(count)