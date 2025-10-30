# Read the three-digit integer N as a string to easily access its digits
N_str = input()

# Extract the hundreds, tens, and ones digits
# a is the hundreds digit
# b is the tens digit
# c is the ones digit
a = int(N_str[0])
b = int(N_str[1])
c = int(N_str[2])

# Form the first integer by arranging b, c, a
# We can concatenate their string representations and then convert to an integer
num1_str = str(b) + str(c) + str(a)
num1 = int(num1_str)

# Form the second integer by arranging c, a, b
num2_str = str(c) + str(a) + str(b)
num2 = int(num2_str)

# Print the two integers separated by a space
print(num1, num2)