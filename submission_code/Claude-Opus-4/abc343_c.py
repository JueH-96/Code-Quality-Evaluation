# YOUR CODE HERE
def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

N = int(input())

# Find the maximum x such that x^3 <= N
# Using binary search would be more efficient, but simple approach works too
max_x = int(N ** (1/3))

# Handle potential floating point errors
while (max_x + 1) ** 3 <= N:
    max_x += 1
while max_x ** 3 > N:
    max_x -= 1

# Search from max_x down to 1
for x in range(max_x, 0, -1):
    cube = x ** 3
    if is_palindrome(cube):
        print(cube)
        break