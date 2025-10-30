# Read input values
N, T, A = map(int, input().split())

# Calculate remaining votes
remaining = N - (T + A)

# Determine if the outcome is already decided
if T > A + remaining:
    print("Yes")
elif A > T + remaining:
    print("Yes")
else:
    print("No")