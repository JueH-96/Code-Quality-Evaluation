# Read the input values
A, B, C = map(int, input().split())

# Determine if Takahashi can shout his love for takoyaki
if B < C:
    # Takahashi sleeps and wakes up on the same day
    can_shout = B < A < C or B < A + 24 < C
else:
    # Takahashi sleeps through midnight
    can_shout = not (C <= A < B)

# Print the result
print("Yes" if can_shout else "No")