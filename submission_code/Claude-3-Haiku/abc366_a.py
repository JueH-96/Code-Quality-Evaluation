# Read the input
n, t, a = map(int, input().split())

# Calculate the remaining votes
remaining_votes = n - (t + a)

# Check if the outcome is already decided
if t > (n // 2):
    print("Yes")
else:
    print("No")