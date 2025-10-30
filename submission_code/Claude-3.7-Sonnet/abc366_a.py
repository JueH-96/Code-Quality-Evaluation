# Read input
N, T, A = map(int, input().split())

# Calculate remaining votes
remaining_votes = N - (T + A)

# Check if outcome is decided
if T > A and T > A + remaining_votes:
    # Takahashi is ahead and will stay ahead even if all remaining votes go to Aoki
    print("Yes")
elif A > T and A > T + remaining_votes:
    # Aoki is ahead and will stay ahead even if all remaining votes go to Takahashi
    print("Yes")
else:
    # Outcome not decided yet
    print("No")