def find_minimum_x(N, A, B):
    A.sort()
    B.sort()
    
    # Two pointers for toys and boxes
    toy_index = 0
    box_index = 0
    
    # Track the largest toy that cannot be placed in any existing box
    largest_unplaced_toy = -1
    
    while toy_index < N:
        if box_index < N - 1 and B[box_index] >= A[toy_index]:
            # The current box can accommodate the current toy
            box_index += 1
            toy_index += 1
        else:
            # The current toy cannot be placed in any of the remaining boxes
            largest_unplaced_toy = max(largest_unplaced_toy, A[toy_index])
            toy_index += 1
    
    # If all toys can be placed in the existing boxes, no need for a new box
    if largest_unplaced_toy == -1:
        return -1
    
    # The minimum size of the new box should be at least the size of the largest unplaced toy
    return largest_unplaced_toy

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N]))

# Output the result
print(find_minimum_x(N, A, B))