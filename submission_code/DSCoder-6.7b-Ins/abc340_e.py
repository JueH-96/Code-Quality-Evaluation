def solve(N, M, A, B):
    # Initialize the number of balls in each box
    boxes = [0]*N
    for i in range(N):
        boxes[i] = A[i]

    # Perform the operations
    for i in range(M):
        # Set C to 0
        C = 0

        # Take out all the balls from box B_i and hold them in hand
        balls = boxes[B[i]]
        boxes[B[i]] = 0

        # While holding at least one ball in hand
        while balls > 0:
            # Increase the value of C by 1
            C += 1

            # Put one ball from hand into box (B_i+C)%N
            boxes[(B[i]+C)%N] += 1

            # Decrease the number of balls in hand
            balls -= 1

    # Return the number of balls in each box
    return boxes

# Read the inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Solve the problem
boxes = solve(N, M, A, B)

# Write the answer to stdout
print(' '.join(map(str, boxes)))