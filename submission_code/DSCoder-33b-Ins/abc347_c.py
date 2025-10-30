# YOUR CODE HERE

# Read the inputs
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

# Calculate the total number of days in a week
total_days = A + B

# Check if all plans can be scheduled on holidays
for i in range(N):
    # If the plan is scheduled on a weekday, it cannot be scheduled on a holiday
    if D[i] % total_days > A:
        print("No")
        break
else:
    print("Yes")