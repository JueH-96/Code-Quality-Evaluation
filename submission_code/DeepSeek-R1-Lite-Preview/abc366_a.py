# Read input values
N, T, A = map(int, input().split())

# Calculate remaining votes
remaining = N - (T + A)

# Determine the outcome
if T > A:
    if T > A + remaining:
        print("Yes")
    else:
        print("No")
elif A > T:
    if A > T + remaining:
        print("Yes")
    else:
        print("No")
else:
    # T == A
    print("No")