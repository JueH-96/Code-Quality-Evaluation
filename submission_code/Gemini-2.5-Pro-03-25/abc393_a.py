# YOUR CODE HERE
import sys

# Read input from stdin
# The input consists of two strings S_1 and S_2, separated by a space.
# S_1 indicates Takahashi's status (sick/fine).
# S_2 indicates Aoki's status (sick/fine).
line = sys.stdin.readline().split()
s1 = line[0]
s2 = line[1]

# Takahashi ate oysters {1, 2}.
# Aoki ate oysters {1, 3}.
# Exactly one oyster type among {1, 2, 3, 4} causes sickness.

# We determine the problematic oyster based on the four possible combinations of S_1 and S_2.

# Case 1: Takahashi is sick, Aoki is sick.
# If Takahashi is sick, the bad oyster is either 1 or 2.
# If Aoki is sick, the bad oyster is either 1 or 3.
# For both to be sick, the bad oyster must be common to both sets {1, 2} and {1, 3}.
# The only common oyster is 1.
if s1 == "sick" and s2 == "sick":
    print(1)

# Case 2: Takahashi is sick, Aoki is fine.
# If Takahashi is sick, the bad oyster is either 1 or 2.
# If Aoki is fine, the bad oyster is NOT 1 and NOT 3.
# Combining these conditions: (bad is 1 or 2) AND (bad is not 1) AND (bad is not 3).
# This implies the bad oyster must be 2.
elif s1 == "sick" and s2 == "fine":
    print(2)

# Case 3: Takahashi is fine, Aoki is sick.
# If Takahashi is fine, the bad oyster is NOT 1 and NOT 2.
# If Aoki is sick, the bad oyster is either 1 or 3.
# Combining these conditions: (bad is not 1) AND (bad is not 2) AND (bad is 1 or 3).
# This implies the bad oyster must be 3.
elif s1 == "fine" and s2 == "sick":
    print(3)

# Case 4: Takahashi is fine, Aoki is fine.
# If Takahashi is fine, the bad oyster is NOT 1 and NOT 2.
# If Aoki is fine, the bad oyster is NOT 1 and NOT 3.
# Combining these conditions: (bad is not 1) AND (bad is not 2) AND (bad is not 3).
# Since the bad oyster must be one of {1, 2, 3, 4}, the only remaining possibility is 4.
# This case is covered by the final `else` branch.
else: # This implicitly covers s1 == "fine" and s2 == "fine"
    print(4)