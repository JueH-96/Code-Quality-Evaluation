# Read the input values
S_AB, S_AC, S_BC = input().split()

# Determine the relative ages
# We will represent the ages as a list where the index corresponds to the brother:
# 0: A, 1: B, 2: C
# We will assign ages such that the order is consistent with the given inequalities.

# Initialize the age list with arbitrary values
ages = [0, 0, 0]

# Assign ages based on the inequalities
if S_AB == '<':
    ages[1] += 1
else:
    ages[0] += 1

if S_AC == '<':
    ages[2] += 1
else:
    ages[0] += 1

if S_BC == '<':
    ages[2] += 1
else:
    ages[1] += 1

# Now, determine the order based on the ages
# The brother with the highest age is the oldest, and so on.
# We will create a list of tuples (age, brother) and sort it.
brothers = ['A', 'B', 'C']
age_brother = list(zip(ages, brothers))
age_brother.sort()

# The middle brother is the second in the sorted list
middle_brother = age_brother[1][1]

# Print the result
print(middle_brother)