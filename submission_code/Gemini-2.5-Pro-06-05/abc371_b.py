# YOUR CODE HERE
import sys

def solve():
    """
    Reads baby information chronologically and determines for each
    if they are named Taro based on the kingdom's rules.
    """
    # Read the total number of families (N) and babies (M).
    # The problem statement guarantees a valid input format.
    try:
        n, m = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Gracefully exit if the input stream is empty or malformed.
        return

    # A set to keep track of families that have already had their first son.
    # The first male child born in a family is the eldest son and is named Taro.
    # Using a set provides O(1) average time complexity for checking membership
    # and adding new elements, making the overall solution very efficient.
    families_with_eldest_son = set()

    # Iterate through the M babies' information, which is given in
    # chronological order of birth.
    for _ in range(m):
        # Read the family ID (an integer) and gender (a character 'M' or 'F').
        line = sys.stdin.readline()
        if not line.strip():
            continue # Skip empty lines

        family_id_str, gender = line.split()
        family_id = int(family_id_str)

        # A baby is named Taro if and only if:
        # 1. The baby is male (gender == 'M').
        # 2. They are the first male child in their family. We can check this by
        #    seeing if their family_id has already been added to our set.
        if gender == 'M' and family_id not in families_with_eldest_son:
            # If both conditions are met, the baby is named Taro.
            print("Yes")
            # We then add the family_id to the set to mark that this family
            # has now had its eldest son.
            families_with_eldest_son.add(family_id)
        else:
            # If the baby is female or not the first male child, they are not named Taro.
            print("No")

# Execute the main logic of the program.
solve()