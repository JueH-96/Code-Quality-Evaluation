# Read the input line and split it into strings
input_strings = input().split()

# Convert the strings to integers and store them in a list
numbers = [int(s) for s in input_strings]

# Sort the list of numbers in non-decreasing order
numbers.sort()

# Let the sorted numbers be s0, s1, s2
# Check if s0 * s1 == s2
if numbers[0] * numbers[1] == numbers[2]:
    print("Yes")
else:
    print("No")