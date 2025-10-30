# Read the three integers A, B, and C from standard input
a, b, c = map(int, input().split())

# Determine if Takahashi is awake at time A.
# Takahashi goes to bed at B o'clock and wakes up at C o'clock.
# The sleep interval is the time period from B (inclusive) up to C (exclusive).

# We need to consider two cases for the sleep interval based on the relationship between B and C.
# Since A, B, and C are pairwise different, B will never be equal to C.

# Case 1: B < C. The sleep interval is a single continuous block of time from B to C within the same day.
# This interval is represented as [B, C).
# Takahashi is awake at time A if A is NOT in this sleep interval.
# A is NOT in [B, C) if A is before B (A < B) OR A is at or after C (A >= C).

# Case 2: B > C. The sleep interval wraps around midnight.
# It consists of two parts: from B to the end of the day [B, 24), and from the start of the day to C [0, C).
# The combined sleep interval is [B, 24) U [0, C).
# Takahashi is awake at time A if A is NOT in this combined sleep interval.
# A is NOT in [B, 24) means A < B.
# A is NOT in [0, C) means A >= C.
# So, A is awake if A is NOT (A >= B OR A < C), which is equivalent to (NOT A >= B) AND (NOT A < C),
# which simplifies to A < B AND A >= C.

is_awake = False

if b < c:
    # Case 1: Sleep interval [B, C)
    # Awake if A is before B or A is at or after C
    if a < b or a >= c:
        is_awake = True
else: # This implies b > c because B and C are different
    # Case 2: Sleep interval wraps around [B, 24) U [0, C)
    # Awake if A is at or after C AND A is before B
    if a >= c and a < b:
        is_awake = True

# Print "Yes" if Takahashi is awake at time A, otherwise print "No".
if is_awake:
    print("Yes")
else:
    print("No")