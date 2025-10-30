import sys

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Sort the candies and boxes in ascending order
A.sort()
B.sort()

# Initialize the total cost to 0
total_cost = 0

# If the number of boxes is less than the number of people, it's impossible to satisfy the condition
if M > N:
    print(-1)
else:
    # For each person, find the box with the minimum cost that satisfies the condition
    for i in range(M):
        # If the box with the minimum cost doesn't have enough candies, it's impossible to satisfy the condition
        if A[i] < B[i]:
            print(-1)
            sys.exit()
        # Add the cost of the box to the total cost
        total_cost += A[i]

    # Print the total cost
    print(total_cost)