import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
S = data[2]

# Convert the string S to a list for easier manipulation
boxes = list(S)

# Count the number of cookies eaten
cookies_eaten = 0

# Iterate from the end of the list to the beginning
for i in range(N-1, -1, -1):
    if boxes[i] == '@':
        cookies_eaten += 1
        if cookies_eaten <= D:
            boxes[i] = '.'
        else:
            break

# Convert the list back to a string
result = ''.join(boxes)

# Print the result
sys.stdout.write(result + '
')