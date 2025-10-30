import sys

# Read the integer X from standard input
x = int(sys.stdin.readline())

# Calculate the ceiling of X / 10 using integer arithmetic.
# For any integer a and positive integer b, ceil(a / b) can be calculated as:
# a // b if a is a multiple of b (a % b == 0)
# a // b + 1 if a is not a multiple of b (a % b != 0)
# This can be concisely written as a // b + (a % b != 0),
# where (a % b != 0) evaluates to 1 if the condition is true, and 0 otherwise,
# leveraging Python's boolean-to-integer conversion.
# This works correctly because Python's % operator for a negative a and positive b
# gives a result r such that 0 <= r < b.
# In this problem, b = 10.
result = x // 10 + (x % 10 != 0)

# Print the result to standard output
print(result)