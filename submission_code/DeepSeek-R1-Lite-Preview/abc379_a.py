# Read the input as a string
N_str = input().strip()

# Extract digits a, b, c
a = int(N_str[0])
b = int(N_str[1])
c = int(N_str[2])

# Form the two new integers
first_num = b * 100 + c * 10 + a
second_num = c * 100 + a * 10 + b

# Print the result
print(first_num, second_num)