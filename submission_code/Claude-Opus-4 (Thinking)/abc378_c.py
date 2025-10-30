# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Dictionary to store the most recent position of each element
last_position = {}

# Result array
b = []

for i in range(n):
    if a[i] in last_position:
        # Element has appeared before
        b.append(last_position[a[i]])
    else:
        # Element hasn't appeared before
        b.append(-1)
    
    # Update the last position of this element
    # Positions are 1-indexed in the problem
    last_position[a[i]] = i + 1

print(' '.join(map(str, b)))