# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# Initialize the list to store the sum of steps for each week
B = []

# Iterate over the days in steps of 7 to calculate the sum for each week
for i in range(0, 7 * N, 7):
    # Calculate the sum of steps for the current week
    week_sum = sum(A[i:i+7])
    # Append the sum to the list
    B.append(week_sum)

# Print the results separated by spaces
print(' '.join(map(str, B)))