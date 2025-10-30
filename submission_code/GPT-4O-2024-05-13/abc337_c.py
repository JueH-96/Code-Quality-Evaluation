# YOUR CODE HERE
def find_line_order(N, A):
    # Create an array to store the position of each person
    position = [-1] * N
    
    # Find the person at the front
    front = A.index(-1) + 1
    
    # Initialize the current person to the front person
    current_person = front
    
    # Fill the position array
    for i in range(N):
        position[i] = current_person
        # Move to the next person in the line
        if A[current_person - 1] != -1:
            current_person = A[current_person - 1]
        else:
            break
    
    return position

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Find the order of people in the line
order = find_line_order(N, A)

# Print the order
print(" ".join(map(str, order)))