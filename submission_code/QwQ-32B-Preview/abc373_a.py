# Read the 12 strings from input
strings = [input().strip() for _ in range(12)]

# Initialize a counter to keep track of matches
count = 0

# Iterate through each i from 1 to 12
for i in range(1, 13):
    # Check if the length of the (i-1)th string equals i
    if len(strings[i-1]) == i:
        count += 1

# Print the final count
print(count)