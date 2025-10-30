# YOUR CODE HERE
s_ab, s_ac, s_bc = input().split()

# Calculate the "older than others" score for each brother.
# This score represents how many other brothers each person is older than.
# In Python, True evaluates to 1 and False evaluates to 0 in arithmetic operations.
# A is older than B if s_ab is '>', A is older than C if s_ac is '>'.
score_a = (s_ab == '>') + (s_ac == '>')

# B is older than A if A is younger than B (s_ab is '<'), B is older than C if s_bc is '>'.
score_b = (s_ab == '<') + (s_bc == '>')

# C is older than A if A is younger than C (s_ac is '<'), C is older than B if B is younger than C (s_bc is '<').
score_c = (s_ac == '<') + (s_bc == '<')

# With three distinct ages and no contradictions, the scores for the three brothers
# will be 0, 1, and 2 in some order.
# The person with a score of 0 is the youngest (older than 0 others).
# The person with a score of 1 is the middle (older than 1 other).
# The person with a score of 2 is the oldest (older than 2 others).
# We are looking for the middle brother, which is the one with a score of 1.

if score_a == 1:
    print("A")
elif score_b == 1:
    print("B")
else:
    # If score_a is not 1 and score_b is not 1, then score_c must be 1
    # because the scores must be 0, 1, 2.
    print("C")