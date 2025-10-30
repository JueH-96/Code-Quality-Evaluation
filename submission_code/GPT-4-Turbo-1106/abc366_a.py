# Read the inputs from stdin
N, T, A = map(int, input().split())

# Calculate the remaining votes
remaining_votes = N - (T + A)

# Determine if the outcome is already decided
if T > A + remaining_votes or A > T + remaining_votes:
    print("Yes")
else:
    print("No")