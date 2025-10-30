# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, processes baby information chronologically, 
    and determines if each baby is named Taro based on the rules.
    Prints the result for each baby to standard output.
    """
    
    # Read N (number of families) and M (number of babies) from the first line of input.
    # map(int, ...) applies the int() function to each element ('N' and 'M' as strings)
    # returned by sys.stdin.readline().split(), which splits the line by whitespace.
    n, m = map(int, sys.stdin.readline().split())
    
    # Initialize a boolean list to track whether the eldest son has been born for each family.
    # `has_eldest_son[i]` will be True if family `i` has already had its first male baby (the eldest son).
    # The list size is N+1 to accommodate 1-based indexing for family IDs (from 1 to N).
    # Index 0 is unused. All entries are initially False, as no babies have been born yet.
    has_eldest_son = [False] * (n + 1) 
    
    # Process the information for each of the M babies.
    # The input guarantees that the baby information is provided in chronological order of birth.
    for _ in range(m):
        # Read the line containing the current baby's family ID (A_i) and gender (B_i).
        line = sys.stdin.readline().split()
        # Convert the family ID from a string to an integer.
        family_id = int(line[0])
        # Get the gender as a string ('M' for male, 'F' for female).
        gender = line[1]
        
        # Assume by default that the current baby is not named Taro.
        is_taro = False
        
        # Check if the conditions for being named Taro are met.
        # Rule: A baby is named Taro if and only if they are the eldest son in their family.
        # The eldest son is the earliest born male child in that family.
        
        # Condition 1: The baby must be male.
        if gender == 'M':
            # Condition 2: This baby must be the *first* male child born in this specific family.
            # We check this using our tracking list `has_eldest_son`.
            # If `has_eldest_son[family_id]` is False, it means no male baby (eldest son) 
            # has been recorded for this family yet.
            if not has_eldest_son[family_id]:
                # Both conditions are met. This baby is the eldest son and should be named Taro.
                is_taro = True
                # Update the tracking list: mark that family `family_id` has now had its eldest son.
                # This prevents any subsequent male babies born into the same family from being named Taro.
                has_eldest_son[family_id] = True
        
        # Print the result ("Yes" or "No") for the current baby to standard output.
        if is_taro:
            print("Yes")
        else:
            print("No")

# Call the main function `solve()` to execute the program logic when the script runs.
solve()