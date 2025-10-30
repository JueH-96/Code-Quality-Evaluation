# Read the input relationships for AB, AC, BC
s_ab, s_ac, s_bc = input().split()

# Initialize a dictionary to store the number of brothers each person is older than.
# This count can be thought of as "wins" in an age comparison.
# For example, if A is older than B (A > B), then A "wins" against B.
# If A is younger than B (A < B), then B "wins" against A.
wins = {'A': 0, 'B': 0, 'C': 0}

# Process the relationship between A and B (S_AB)
if s_ab == '>':  # A is older than B
    wins['A'] += 1
else:  # A is younger than B (s_ab must be '<')
    wins['B'] += 1

# Process the relationship between A and C (S_AC)
if s_ac == '>':  # A is older than C
    wins['A'] += 1
else:  # A is younger than C (s_ac must be '<')
    wins['C'] += 1

# Process the relationship between B and C (S_BC)
if s_bc == '>':  # B is older than C
    wins['B'] += 1
else:  # B is younger than C (s_bc must be '<')
    wins['C'] += 1

# Determine the middle brother.
# The youngest brother is older than 0 others (0 wins).
# The middle brother is older than 1 other (1 win).
# The oldest brother is older than 2 others (2 wins).
# We are looking for the brother with exactly 1 win.

middle_brother_name = None
# Iterate through the brothers to find the one with 1 win.
# The order of iteration ('A', 'B', 'C') is arbitrary but covers all possibilities.
for brother_name in ['A', 'B', 'C']:
    if wins[brother_name] == 1:
        middle_brother_name = brother_name
        # Since there's guaranteed to be exactly one middle brother,
        # we can stop searching once found.
        break

# Print the name of the middle brother
print(middle_brother_name)