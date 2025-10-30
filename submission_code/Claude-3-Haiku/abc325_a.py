import sys

# Read input from stdin
input_line = sys.stdin.readline().strip().split()
surname = input_line[0]
first_name = input_line[1]

# Concatenate the surname, a space, and the honorific "san"
output = f"{surname} san"

# Print the output to stdout
print(output)