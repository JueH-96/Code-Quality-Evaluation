# YOUR CODE HERE
import sys

# Read the input three-digit integer N as a string
n_str = sys.stdin.readline().strip()

# The input N is guaranteed to be a three-digit integer.
# Let a, b, c be the hundreds, tens, and ones digits respectively.
# In the string representation n_str:
# n_str[0] corresponds to the hundreds digit 'a'
# n_str[1] corresponds to the tens digit 'b'
# n_str[2] corresponds to the ones digit 'c'

# Extract the digits as characters
a_char = n_str[0]
b_char = n_str[1]
c_char = n_str[2]

# The first required integer is formed by arranging the digits b, c, a.
# Construct the string representation of this integer.
num1_str = b_char + c_char + a_char

# The second required integer is formed by arranging the digits c, a, b.
# Construct the string representation of this integer.
num2_str = c_char + a_char + b_char

# Convert the string representations to integers.
num1 = int(num1_str)
num2 = int(num2_str)

# Print the two integers separated by a space.
print(f"{num1} {num2}")

# Alternatively, using integer arithmetic:
# n = int(n_str)
# a = n // 100
# b = (n // 10) % 10
# c = n % 10
#
# num1 = b * 100 + c * 10 + a
# num2 = c * 100 + a * 10 + b
#
# print(f"{num1} {num2}")
# Both methods yield the same result. The string manipulation method is used here.
# YOUR CODE HERE