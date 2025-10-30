# YOUR CODE HERE
N, D = map(int, input().split())
S = input().strip()

# Convert string to list for easier manipulation
boxes = list(S)

# For D days, eat cookies from rightmost box
for day in range(D):
    # Find the rightmost box with a cookie
    for i in range(N-1, -1, -1):
        if boxes[i] == '@':
            boxes[i] = '.'
            break

# Convert back to string and print
result = ''.join(boxes)
print(result)