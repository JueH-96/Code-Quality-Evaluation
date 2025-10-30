import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Convert the first element to integer N
N = int(data[0])

# Extract the strings from index 1 to N (inclusive)
strings = data[1:N+1]

# Sort the strings in ascending order of length
sorted_strings = sorted(strings, key=len)

# Concatenate the sorted strings and print the result
print(''.join(sorted_strings))