# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
W = list(map(int, data[N+1:2*N+1]))

# Create a list of tuples (weight, box)
items = list(zip(W, A))

# Sort items by weight
items.sort()

# Create a list to keep track of the boxes
boxes = [0] * (N + 1)

# Initialize the total cost
total_cost = 0

# Iterate over the sorted items
for weight, box in items:
    # Find the first empty box
    for i in range(1, N + 1):
        if boxes[i] == 0:
            # Move the item to the empty box
            boxes[i] = box
            total_cost += weight
            break

# Print the total cost
print(total_cost)