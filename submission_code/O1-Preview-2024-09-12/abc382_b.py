# YOUR CODE HERE
# Read inputs
N, D = map(int, input().split())
S = input()

# Convert S to a list for easy manipulation
boxes = list(S)

for _ in range(D):
    # Find the rightmost box with a cookie
    for i in range(N - 1, -1, -1):
        if boxes[i] == '@':
            boxes[i] = '.'
            break  # Exit the loop after removing one cookie

# Convert the list back to a string and print
print(''.join(boxes))