# Read input values for A, B, C
# A: takoyaki time
# B: bedtime
# C: wake-up time
A, B, C = map(int, input().split())

# Determine if Takahashi is awake at time A.

if B < C:
    # Case 1: Sleep period does not cross midnight (e.g., B=8, C=14).
    # Takahashi sleeps from B (inclusive) to C (exclusive).
    # He is awake if A is before B, or A is C or later.
    if A < B or A >= C:
        print("Yes")
    else:
        # A is in [B, C), so he is asleep.
        print("No")
else:  # B > C
    # Case 2: Sleep period crosses midnight (e.g., B=21, C=7).
    # Takahashi sleeps from B (inclusive) to 24 (exclusive) on one day,
    # and from 0 (inclusive) to C (exclusive) on the next day.
    # He is awake if A is from C (inclusive) to B (exclusive).
    if C <= A and A < B:
        print("Yes")
    else:
        # A is in [B, 24) or [0, C), so he is asleep.
        print("No")