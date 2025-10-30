# YOUR CODE HERE
Q = int(input())
stack = [0] * 100  # Initialize with 100 cards labeled 0

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        # Type 1: Push a card with value x
        x = int(query[1])
        stack.append(x)
    else:
        # Type 2: Pop and print the top card
        print(stack.pop())