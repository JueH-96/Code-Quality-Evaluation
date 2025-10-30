import sys

# Read input
N, M = map(int, input().split())
B = list(map(int, input().split()))
W = list(map(int, input().split()))

# Sort the black and white balls in descending order
B.sort(reverse=True)
W.sort(reverse=True)

# Initialize the maximum sum and the number of black balls chosen
max_sum = 0
black_chosen = 0

# Iterate through the black balls
for b in B:
    # If the number of black balls chosen is greater than or equal to the number of white balls chosen
    if black_chosen >= len(W):
        max_sum += b
        black_chosen += 1
    # If the number of black balls chosen is less than the number of white balls chosen
    elif black_chosen < len(W) and b >= W[black_chosen]:
        max_sum += b
        black_chosen += 1
    # If the current black ball is less than the current white ball, we can't choose it
    else:
        break

# Print the maximum sum
print(max_sum)