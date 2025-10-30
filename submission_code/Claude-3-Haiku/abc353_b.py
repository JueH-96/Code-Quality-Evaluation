# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Initialize variables
empty_seats = K
start_count = 0

# Iterate through the groups
for i in range(N):
    # If there are no groups left, start the attraction and end the guidance
    if i == N:
        start_count += 1
        break
    
    # Compare the number of empty seats with the number of people in the group
    if empty_seats < A[i]:
        start_count += 1
        empty_seats = K
    else:
        empty_seats -= A[i]

# Print the answer
print(start_count)