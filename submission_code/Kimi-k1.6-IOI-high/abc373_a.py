# Read the 12 input strings
strings = [input().strip() for _ in range(12)]

# Initialize the count
count = 0

# Check each string's length against its position (1-based index)
for i in range(1, 13):
    if len(strings[i-1]) == i:
        count += 1

# Output the result
print(count)