# YOUR CODE HERE
# Read the three-digit integer N from standard input.
# Reading it as a string makes it easy to access individual digits.
n_str = input()

# Let a, b, c be the hundreds, tens, and ones digits of N.
# In the string n_str, a is at index 0, b is at index 1, and c is at index 2.
a = n_str[0]
b = n_str[1]
c = n_str[2]

# Form the first required number by arranging the digits in the order b, c, a.
# This can be done by concatenating the character strings for the digits.
num_bca = b + c + a

# Form the second required number by arranging the digits in the order c, a, b.
num_cab = c + a + b

# Print the two resulting numbers separated by a space.
# The print() function with multiple arguments separates them with a space by default.
print(num_bca, num_cab)