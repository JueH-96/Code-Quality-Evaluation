# YOUR CODE HERE
import sys

# Read input values for A, B, and C from standard input
# A: The time Takahashi needs to shout (0 <= A < 24)
# B: The time Takahashi goes to bed (0 <= B < 24)
# C: The time Takahashi wakes up (0 <= C < 24)
# A, B, and C are pairwise different.
# All input values are integers.
a, b, c = map(int, sys.stdin.readline().split())

# Determine the relationship between bed time (B) and wake up time (C)
# This determines whether the sleep period crosses midnight or not.

if b < c:
    # Case 1: Bed time and wake up time are on the same day cycle (e.g., bed at 8, wake at 14).
    # Takahashi goes to bed at B o'clock and wakes up at C o'clock on the same day.
    # The sleeping interval is [B, C). Note that time is represented from 0 to 23.
    # Takahashi is asleep if the shout time 'a' is within this interval (inclusive of B, exclusive of C).
    # Check if A falls within the sleeping interval [B, C).
    if b <= a < c:
        # If A is within the sleeping interval, Takahashi is asleep and cannot shout.
        print("No")
    else:
        # If A is outside the sleeping interval [B, C), Takahashi is awake and can shout.
        # This means A is either before B (A < B) or at or after C (A >= C).
        print("Yes")
else: # b > c
    # Case 2: Bed time is on one day, wake up time is on the next day (e.g., bed at 21, wake at 7).
    # Takahashi goes to bed at B o'clock on one day and wakes up at C o'clock on the next day.
    # The sleeping interval spans across midnight. It includes times from B to 23 on the first day,
    # and times from 0 to C-1 on the second day. In a single 24-hour cycle, this corresponds
    # to two intervals: [B, 24) and [0, C).
    # Consequently, Takahashi is awake during the interval [C, B).
    # Check if the shout time A falls within the awake interval [C, B).
    if c <= a < b:
        # If A is within the awake interval [C, B), Takahashi can shout.
        print("Yes")
    else:
        # If A is outside the awake interval, it means A falls within the sleeping intervals
        # [B, 24) or [0, C). In this case, Takahashi is asleep and cannot shout.
        print("No")