# YOUR CODE HERE
N, T, A = map(int, input().split())

# Calculate the remaining votes
remaining_votes = N - (T + A)

# Check if Takahashi can still win
if T + remaining_votes > A:
    print("Yes")
else:
    print("No")