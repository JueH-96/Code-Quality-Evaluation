# Read the input from stdin
N, T, A = map(int, input().split())

# Calculate the remaining votes
remaining_votes = N - T - A

# Check if Takahashi's victory is decided
if T + remaining_votes > A:
    print("Yes")
else:
    print("No")