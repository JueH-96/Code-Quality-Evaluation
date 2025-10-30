import sys

input = sys.stdin.read
data = input().strip()

# Extract digits
a = int(data[0])
b = int(data[1])
c = int(data[2])

# Form the new numbers
first_number = b * 100 + c * 10 + a
second_number = c * 100 + a * 10 + b

# Print the results
print(first_number, second_number)