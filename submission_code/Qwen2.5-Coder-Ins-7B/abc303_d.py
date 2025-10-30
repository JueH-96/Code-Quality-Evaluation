# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
X = int(data[0])
Y = int(data[1])
Z = int(data[2])
S = data[3]

# Initialize variables
n = len(S)
ans = 0
caps_lock_on = False

# Iterate through the string S
for char in S:
    if char == 'A':
        if caps_lock_on:
            ans += X
        else:
            ans += Y
    else:
        if caps_lock_on:
            ans += Y
        else:
            ans += X
    # Toggle Caps Lock if the current character is 'A'
    if char == 'A':
        caps_lock_on = not caps_lock_on

# Print the total time required
print(ans)