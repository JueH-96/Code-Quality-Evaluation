# Read the number of weeks
N = int(input())
# Read the list of steps
A = list(map(int, input().split()))
# Initialize a list to store the weekly sums
B = []
# Iterate over each week
for i in range(N):
    # Calculate the sum of steps for the current week
    weekly_sum = sum(A[i*7 : (i+1)*7])
    B.append(weekly_sum)
# Print the result separated by spaces
print(' '.join(map(str, B)))