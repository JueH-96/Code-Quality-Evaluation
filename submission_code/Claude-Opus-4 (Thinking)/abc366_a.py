# YOUR CODE HERE
N, T, A = map(int, input().split())

# Calculate votes needed to win (majority)
votes_to_win = (N + 1) // 2

# Calculate remaining votes
remaining = N - T - A

# Check if outcome is decided
if T >= votes_to_win or A >= votes_to_win:
    # Someone already won
    print("Yes")
elif T + remaining < votes_to_win or A + remaining < votes_to_win:
    # Someone can't possibly win
    print("Yes")
else:
    # Both can still win
    print("No")