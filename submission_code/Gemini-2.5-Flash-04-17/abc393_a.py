# YOUR CODE HERE
import sys

# Read the input line which contains S_1 and S_2 separated by a space
# S_1: Takahashi's status ("sick" or "fine")
# S_2: Aoki's status ("sick" or "fine")
s1, s2 = sys.stdin.readline().split()

# Determine the problematic oyster type (1, 2, 3, or 4)
# Takahashi ate oysters 1 and 2.
# Aoki ate oysters 1 and 3.
# Exactly one oyster type from {1, 2, 3, 4} causes stomach trouble.

# We analyze the four possible combinations of sickness statuses to deduce the cause.

# Case 1: Takahashi is sick (S_1 == "sick") and Aoki is sick (S_2 == "sick")
# - Takahashi got sick from eating {1, 2}. This implies the problematic oyster is 1 or 2.
# - Aoki got sick from eating {1, 3}. This implies the problematic oyster is 1 or 3.
# - For both to be sick, the problematic oyster must be in the intersection of {1, 2} and {1, 3}, which is {1}.
# - If oyster 1 is the cause, both would get sick, matching the observation.
# - If oyster 2, 3, or 4 were the cause, the outcomes would be different (only T sick, only A sick, or neither sick, respectively).
# Conclusion: Oyster 1 is the cause.
if s1 == "sick" and s2 == "sick":
    print(1)

# Case 2: Takahashi is sick (S_1 == "sick") and Aoki is fine (S_2 == "fine")
# - Takahashi got sick from eating {1, 2}. Problematic oyster is 1 or 2.
# - Aoki is fine after eating {1, 3}. This means oysters 1 and 3 are NOT the cause.
# - Combining the deductions: The problematic oyster is (1 or 2) AND (NOT 1) AND (NOT 3).
# - This logical combination simplifies to: Problematic oyster is 2.
# - Check: If 2 is the cause, Takahashi (ate 1, 2) gets sick. Aoki (ate 1, 3) is fine. This matches the observation.
# Conclusion: Oyster 2 is the cause.
elif s1 == "sick" and s2 == "fine":
    print(2)

# Case 3: Takahashi is fine (S_1 == "fine") and Aoki is sick (S_2 == "sick")
# - Takahashi is fine after eating {1, 2}. This means oysters 1 and 2 are NOT the cause.
# - Aoki got sick from eating {1, 3}. Problematic oyster is 1 or 3.
# - Combining the deductions: The problematic oyster is (NOT 1) AND (NOT 2) AND (1 or 3).
# - This logical combination simplifies to: Problematic oyster is 3.
# - Check: If 3 is the cause, Takahashi (ate 1, 2) is fine. Aoki (ate 1, 3) gets sick. This matches the observation.
# Conclusion: Oyster 3 is the cause.
elif s1 == "fine" and s2 == "sick":
    print(3)

# Case 4: Takahashi is fine (S_1 == "fine") and Aoki is fine (S_2 == "fine")
# - Takahashi is fine after eating {1, 2}. This means oysters 1 and 2 are NOT the cause.
# - Aoki is fine after eating {1, 3}. This means oysters 1 and 3 are NOT the cause.
# - Thus, we've deduced that oysters 1, 2, and 3 are NOT the cause.
# - The problem states that exactly one oyster type from {1, 2, 3, 4} is the cause.
# - Since 1, 2, and 3 are not the cause, the cause must be 4.
# - Check: If 4 is the cause, Takahashi (ate 1, 2) is fine. Aoki (ate 1, 3) is fine. This matches the observation.
# Conclusion: Oyster 4 is the cause.
elif s1 == "fine" and s2 == "fine":
    print(4)

# The problem guarantees that S_1 and S_2 will be exactly "sick" or "fine",
# ensuring that one of these four conditions will always be met.
# The logic based on who ate what and who got sick correctly identifies the
# unique problematic oyster type in each scenario according to the problem constraints.