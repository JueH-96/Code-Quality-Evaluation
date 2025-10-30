# Read the 12 strings
strings = [input().strip() for _ in range(12)]

# Initialize the count
count = 0

# Iterate through each string and check the condition
for i in range(12):
    if len(strings[i]) == i + 1:
        count += 1

# Print the result
print(count)