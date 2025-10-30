# Read N and M from the input
N, M = map(int, input().split())

# Read the initial number of balls in each box
A = list(map(int, input().split()))

# Read the sequence of operations
B = list(map(int, input().split()))

# Perform the operations
for b in B:
    # Set C to 0
    C = 0
    # Take out all the balls from box B_i
    balls_in_hand = A[b]
    A[b] = 0
    # While holding at least one ball in hand, repeat the process
    while balls_in_hand > 0:
        # Increase the value of C by 1
        C += 1
        # Calculate the index of the box to put a ball into
        box_index = (b + C) % N
        # Put one ball from hand into the box
        A[box_index] += 1
        # Decrease the number of balls in hand
        balls_in_hand -= 1

# Print the number of balls in each box
print(' '.join(map(str, A)))