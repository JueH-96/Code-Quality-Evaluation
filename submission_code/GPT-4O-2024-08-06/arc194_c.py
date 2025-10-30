# YOUR CODE HERE
def min_cost_to_transform(N, A, B, C):
    # Calculate initial cost of A
    initial_cost = sum(A[i] * C[i] for i in range(N))
    
    # List to store the cost difference for each flip
    cost_differences = []
    
    # Calculate the cost difference for each position where A[i] != B[i]
    for i in range(N):
        if A[i] != B[i]:
            # If we flip A[i], the cost changes by this amount
            # If A[i] is 0 and we flip to 1, we add C[i] to the cost
            # If A[i] is 1 and we flip to 0, we subtract C[i] from the cost
            if A[i] == 0:
                cost_difference = C[i]
            else:
                cost_difference = -C[i]
            cost_differences.append(cost_difference)
    
    # Sort the cost differences in descending order
    cost_differences.sort(reverse=True)
    
    # Apply the flips and calculate the minimum total cost
    min_total_cost = initial_cost
    for cost_diff in cost_differences:
        min_total_cost += cost_diff
    
    return min_total_cost

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N+1]))
C = list(map(int, data[2*N+1:3*N+1]))

# Calculate and print the result
result = min_cost_to_transform(N, A, B, C)
print(result)