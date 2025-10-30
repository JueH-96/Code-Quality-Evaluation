# Read the input values A, B, and C
A, B, C = map(int, input().split())

# Initialize a flag to determine if Takahashi is asleep at time A
is_asleep_at_A = False

# There are two main scenarios for Takahashi's sleep schedule:
# 1. He goes to bed and wakes up on the same "day" (sleep does not cross midnight).
#    Example: B=8, C=14 (sleeps from 8:00 to 13:59)
# 2. He goes to bed on one day and wakes up on the next day (sleep crosses midnight).
#    Example: B=21, C=7 (sleeps from 21:00 to 06:59)

if B < C:
    # Scenario 1: Sleep is within the same "day" (B < C).
    # Takahashi is asleep if A is between B (inclusive) and C (exclusive).
    # So, if B <= A and A < C, he is asleep.
    if B <= A and A < C:
        is_asleep_at_A = True
else: # B > C (since B and C are different and 0 <= B, C < 24)
    # Scenario 2: Sleep crosses midnight (B > C).
    # Takahashi goes to bed at B and wakes up at C the next day.
    # This means he is asleep during two intervals:
    # 1. From B o'clock until the end of the day (23:59).
    # 2. From the start of the day (00:00) until C o'clock (exclusive).
    # So, if A is greater than or equal to B OR A is less than C, he is asleep.
    if A >= B or A < C:
        is_asleep_at_A = True

# Determine if he can shout (i.e., if he is awake at time A)
if not is_asleep_at_A:
    print("Yes")
else:
    print("No")