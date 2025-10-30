# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort both arrays
A.sort()
B.sort()

# Use two pointers approach
# j will track the current position in A array
j = 0
total_cost = 0

# For each person's requirement (in sorted order)
for i in range(M):
    # Find the first box that has at least B[i] candies
    while j < N and A[j] < B[i]:
        j += 1
    
    # If no such box exists, it's impossible
    if j >= N:
        print(-1)
        exit()
    
    # Use this box for person i
    total_cost += A[j]
    j += 1  # Move to next box

print(total_cost)