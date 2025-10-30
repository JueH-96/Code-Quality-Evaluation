import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the order list
order = []

# Find the starting point (the person at the front of the line)
start = A.index(-1) + 1

# Follow the chain to determine the order
current = start
while current != -1:
    order.append(current)
    next_person = A[current - 1]
    A[current - 1] = -1  # Mark as visited
    current = next_person

# Print the order
print(" ".join(map(str, order)))