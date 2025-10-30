def is_good_string_possible(N, A):
    # Check if there are any zeros in A
    if all(a == 1 for a in A):
        return "Yes"
    
    # Check for the presence of at least one '1' in A
    for i in range(N):
        if A[i] == 0:
            # Check the pattern around the zero
            if A[i-1] == 1 or A[(i+1) % N] == 1 or A[(i-2) % N] == 1 or A[(i+2) % N] == 1:
                return "Yes"
    
    return "No"

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Get the result
result = is_good_string_possible(N, A)

# Print the result
print(result)