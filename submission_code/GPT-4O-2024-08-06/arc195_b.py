def can_make_equal_sums(N, A, B):
    fixed_sum = None
    
    # Determine the fixed sum from non-negative pairs
    for i in range(N):
        if A[i] != -1 and B[i] != -1:
            current_sum = A[i] + B[i]
            if fixed_sum is None:
                fixed_sum = current_sum
            elif fixed_sum != current_sum:
                return "No"
    
    # If no fixed sum was found, we can choose any non-negative sum
    if fixed_sum is None:
        fixed_sum = 0
    
    # Check if we can adjust -1s to match the fixed sum
    for i in range(N):
        if A[i] == -1 and B[i] != -1:
            A[i] = fixed_sum - B[i]
            if A[i] < 0:
                return "No"
        elif B[i] == -1 and A[i] != -1:
            B[i] = fixed_sum - A[i]
            if B[i] < 0:
                return "No"
        elif A[i] == -1 and B[i] == -1:
            # We can set A[i] = 0 and B[i] = fixed_sum or vice versa
            A[i] = 0
            B[i] = fixed_sum
    
    return "Yes"

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N+1]))

# Output the result
print(can_make_equal_sums(N, A, B))