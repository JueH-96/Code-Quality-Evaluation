# YOUR CODE HERE
def min_operations(N, Q, instructions):
    # Initialize positions of left and right hands
    left_hand = 1
    right_hand = 2
    
    # Function to calculate the minimum distance between two parts on the ring
    def min_distance(a, b):
        return min(abs(a - b), N - abs(a - b))
    
    # Initialize the total number of operations
    total_operations = 0
    
    for H, T in instructions:
        if H == 'L':
            # Move left hand to T
            if left_hand != T:
                total_operations += min_distance(left_hand, T)
                left_hand = T
        elif H == 'R':
            # Move right hand to T
            if right_hand != T:
                total_operations += min_distance(right_hand, T)
                right_hand = T
    
    return total_operations

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
instructions = [(data[2 + 2 * i], int(data[3 + 2 * i])) for i in range(Q)]

# Calculate and print the result
result = min_operations(N, Q, instructions)
print(result)