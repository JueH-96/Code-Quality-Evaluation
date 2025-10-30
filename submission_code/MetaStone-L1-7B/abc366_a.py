# Read the input values
N, T, A = map(int, input().split())

# Calculate the remaining votes
remaining = N - T - A

# Check if the outcome is already decided
if T > A + remaining or A > T + remaining:
    print("Yes")
else:
    print("No")