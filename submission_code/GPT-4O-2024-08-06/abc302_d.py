# YOUR CODE HERE
def find_maximum_gift_sum(N, M, D, A, B):
    # Sort both lists
    A.sort()
    B.sort()
    
    # Initialize pointers and max sum
    i, j = 0, 0
    max_sum = -1
    
    # Use two pointers to find the maximum sum
    while i < N and j < M:
        if abs(A[i] - B[j]) <= D:
            # If the condition is satisfied, update the max sum
            max_sum = max(max_sum, A[i] + B[j])
            # Move to the next element in B to try for a larger sum
            j += 1
        elif A[i] < B[j]:
            # If A[i] is less, move i to try a larger value in A
            i += 1
        else:
            # If B[j] is less, move j to try a larger value in B
            j += 1
    
    return max_sum

import sys
input = sys.stdin.read
data = input().split()

# Read inputs
N = int(data[0])
M = int(data[1])
D = int(data[2])
A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:3+N+M]))

# Get the result
result = find_maximum_gift_sum(N, M, D, A, B)

# Print the result
print(result)