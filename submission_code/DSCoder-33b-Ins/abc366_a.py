# YOUR CODE HERE

# Read the inputs
N, T, A = map(int, input().split())

# Calculate the remaining votes needed for Takahashi to win
remaining_votes = N // 2 + 1

# Check if Takahashi has enough votes
if T + remaining_votes > A:
    print("Yes")
else:
    print("No")