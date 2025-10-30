# Read the input from stdin and split it into a list of strings
input_values = input().split()

# Convert the list of strings to a list of integers
values = list(map(int, input_values))

# Create a set from the list to find the number of unique values
unique_values = set(values)

# Check if there are exactly two unique values
if len(unique_values) == 2:
    print("Yes")
else:
    print("No")