N, T, A = map(int, input().split())

# Calculate the remaining votes
remaining_votes = N - (T + A)

# Check if the outcome is decided
if T > A + remaining_votes:
    print("Yes")
else:
    print("No")