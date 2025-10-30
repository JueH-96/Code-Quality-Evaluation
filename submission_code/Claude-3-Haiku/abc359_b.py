# Read the input
N = int(input())
A = list(map(int, input().split()))

# Initialize the count of valid indices
count = 0

# Iterate through the colors from 1 to N
for i in range(1, N+1):
    # Find the indices of the two people wearing clothes of color i
    indices = [j for j in range(2*N) if A[j] == i]
    
    # Check if there is exactly one person between the two people
    if abs(indices[0] - indices[1]) == 2:
        count += 1

# Print the answer
print(count)