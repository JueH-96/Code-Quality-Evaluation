# Read the three-digit integer N as a string
N_str = input()

# Let a, b, c be the hundreds, tens, ones digits of N.
# In terms of string characters:
# a_char is N_str[0]
# b_char is N_str[1]
# c_char is N_str[2]
a_char = N_str[0]
b_char = N_str[1]
c_char = N_str[2]

# The first integer to print is formed by arranging b, c, a in this order.
# This means b is the hundreds digit, c is the tens digit, and a is the ones digit.
# As a string, this is b_char + c_char + a_char.
num1_str = b_char + c_char + a_char

# The second integer to print is formed by arranging c, a, b in this order.
# This means c is the hundreds digit, a is the tens digit, and b is the ones digit.
# As a string, this is c_char + a_char + b_char.
num2_str = c_char + a_char + b_char

# The problem constraints state that each digit is between 1 and 9.
# This means b_char and c_char (which are the leading digits of num1_str and num2_str respectively)
# will not be '0'. Therefore, num1_str and num2_str are direct string representations
# of three-digit integers.
# We need to print these two integers separated by a space.
# The print() function with multiple arguments automatically separates them with a space.
print(num1_str, num2_str)