import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Initialize index and read Q
index = 0
Q = int(data[index])
index += 1

# Initialize the stack with 100 zeros
stack = [0] * 100

# Process each query
for _ in range(Q):
    c = int(data[index])
    index += 1
    if c == 1:
        # Type 1 query: add a card with value x
        x = int(data[index])
        index += 1
        stack.append(x)
    elif c == 2:
        # Type 2 query: remove the top card and print its value
        val = stack.pop()
        print(val)