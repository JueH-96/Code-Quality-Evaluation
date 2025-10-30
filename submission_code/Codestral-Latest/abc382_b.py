# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
S = data[2]

# Convert the string to a list for easier manipulation
boxes = list(S)

# Count the number of cookies
cookie_count = boxes.count('@')

# Simulate the process for D days
for _ in range(D):
    # Find the rightmost cookie
    rightmost_cookie_index = len(boxes) - 1 - boxes[::-1].index('@')
    # Remove the cookie
    boxes[rightmost_cookie_index] = '.'

# Convert the list back to a string
result = ''.join(boxes)

# Print the result
print(result)