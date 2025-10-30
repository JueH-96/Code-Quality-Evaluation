# Read input strings from stdin
strings = [input() for _ in range(12)]

# Initialize count of matching lengths
count = 0

# Iterate over the strings and their indices
for i, string in enumerate(strings, start=1):
    # Check if the length of the string matches its index
    if len(string) == i:
        count += 1

# Print the count of matching lengths
print(count)