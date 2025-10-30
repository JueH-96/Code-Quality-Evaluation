# YOUR CODE HERE

# Read the inputs
strings = []
for _ in range(12):
    strings.append(input())

# Count the number of strings whose length is equal to their index
count = 0
for i in range(12):
    if len(strings[i]) == i+1:
        count += 1

# Print the result
print(count)