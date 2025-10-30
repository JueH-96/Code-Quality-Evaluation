# YOUR CODE HERE
A, B, C = map(int, input().split())

# Determine if Takahashi is awake at time A
if B < C:
    # Sleeps from B to C in the same day
    is_awake = A < B or A >= C
else:
    # Sleeps from B through midnight until C
    is_awake = C <= A < B

if is_awake:
    print("Yes")
else:
    print("No")