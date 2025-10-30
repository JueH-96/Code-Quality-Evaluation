# YOUR CODE HERE
import sys

# Read N (number of families) and M (number of babies) from the first line of input.
# N and M are space-separated integers.
n, m = map(int, sys.stdin.readline().split())

# Initialize a list `has_first_male` to keep track of whether a family has already
# had its first male child born.
# The list uses 1-based indexing to match family IDs (1 to N).
# `has_first_male[i]` will be True if family `i` has already had its first male baby born,
# and False otherwise.
# Initially, none of the N families have had any babies (and thus no male babies),
# so all entries are initialized to False.
has_first_male = [False] * (n + 1) # Create a list of size N+1 initialized with False values.

# Process information for each of the M babies in the chronological order they are born.
# The problem states that the input gives information about babies in chronological order.
for _ in range(m):
    # Read the family ID and gender for the current baby.
    # Each baby's information is provided on a new line, space-separated.
    line = sys.stdin.readline().split()
    family_id = int(line[0]) # The family ID where the baby was born (1 <= family_id <= N)
    gender = line[1]        # The gender of the baby ('M' for male, 'F' for female)

    # Determine if the current baby is named Taro.
    # According to the rules:
    # - The eldest son is the earliest born male child in each family.
    # - No one else is given the name Taro.
    # Thus, a baby is named Taro if and only if they are:
    # 1. Male.
    # 2. The earliest born male child in their specific family among all babies born so far (including this one).
    # Since we process babies in chronological order, checking if a family has *already* had a male
    # before the current baby tells us if the current male baby is the *first* male in that family.

    # Check condition 1: Is the baby male?
    if gender == 'M':
        # If the baby is male, now check condition 2: Are they the earliest male in their family?
        # We use the `has_first_male` list to check if a previous male baby
        # (earlier in chronological order) has already been born in this family.
        if not has_first_male[family_id]:
            # If `has_first_male[family_id]` is False, it means no male baby
            # has been recorded for this family yet among the babies processed so far.
            # Since the babies are processed in chronological order, this current male baby
            # is indeed the earliest born male child in this family.
            # This baby is named Taro.
            print("Yes")
            # Now that family `family_id` has had its first male baby, we update
            # the status in `has_first_male` to True. Any subsequent male babies
            # born in this family will not be the earliest.
            has_first_male[family_id] = True
        else:
            # If `has_first_male[family_id]` is True, it means a male baby was
            # already processed for this family earlier in the input.
            # That earlier male baby was the first born male in this family.
            # This current male baby is not the earliest and is therefore not named Taro.
            print("No")
    else:
        # If the baby's gender is 'F' (female), they cannot be named Taro according to the rules.
        # Only male babies are considered for the name Taro.
        print("No")