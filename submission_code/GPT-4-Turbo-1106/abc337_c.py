def find_line_order(N, A):
    # Initialize an empty list to store the line order
    line_order = [0] * N
    # Find the index of the person at the front
    front_index = A.index(-1)
    # Set the person at the front in the line order
    line_order[0] = front_index + 1
    
    # Fill the line order from the front to the back
    for i in range(1, N):
        # The person behind the current person is A[front_index] - 1
        line_order[i] = A[front_index]
        # Update the front_index to the next person in line
        front_index = A[front_index] - 1
    
    return line_order

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Find the line order
line_order = find_line_order(N, A)

# Print the line order to stdout
print(' '.join(map(str, line_order)))