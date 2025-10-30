# YOUR CODE HERE
A, B, C = map(int, input().split())

# Determine if A is within the awake period
if B < C:
    # Awake period is from B to C
    can_shout = B <= A < C
else:
    # Awake period wraps around midnight
    can_shout = A >= B or A < C

print("Yes" if can_shout else "No")