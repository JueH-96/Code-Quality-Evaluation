# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read the three comparison strings from standard input, separated by spaces.
    # Example input: "< < <"
    # After split(): ['<', '<', '<']
    s_ab, s_ac, s_bc = sys.stdin.readline().split()

    # Initialize a dictionary to store the number of 'wins' for each brother.
    # A 'win' means being determined as older in a pairwise comparison.
    # For example, if A > B, then A gets a 'win'.
    wins = {'A': 0, 'B': 0, 'C': 0}

    # Process the comparison between A and B based on S_AB
    if s_ab == '<':  # A is younger than B (A < B), so B is older than A. B gets a 'win'.
        wins['B'] += 1
    else:  # s_ab == '>', A is older than B (A > B). A gets a 'win'.
        wins['A'] += 1

    # Process the comparison between A and C based on S_AC
    if s_ac == '<':  # A is younger than C (A < C), so C is older than A. C gets a 'win'.
        wins['C'] += 1
    else:  # s_ac == '>', A is older than C (A > C). A gets a 'win'.
        wins['A'] += 1

    # Process the comparison between B and C based on S_BC
    if s_bc == '<':  # B is younger than C (B < C), so C is older than B. C gets a 'win'.
        wins['C'] += 1
    else:  # s_bc == '>', B is older than C (B > C). B gets a 'win'.
        wins['B'] += 1

    # At this point, the `wins` dictionary contains the count of how many other brothers each brother is older than.
    # The total number of comparisons is 3. Each comparison results in one 'win'.
    # The sum of wins across all brothers will always be 3.
    # Since the problem guarantees there are no contradictions, the ages form a strict total order (e.g., youngest < middle < oldest).
    # In this total order:
    # - The youngest brother is older than 0 others (wins = 0).
    # - The middle brother is older than 1 other (wins = 1).
    # - The oldest brother is older than 2 others (wins = 2).
    
    # We need to find the brother who is the middle one, which corresponds to having exactly 1 win.
    middle_brother = None
    for brother, count in wins.items():
        # Check if the current brother has exactly 1 win
        if count == 1:
            middle_brother = brother
            # Since the problem constraints guarantee a unique middle brother exists,
            # we can stop searching once we find them.
            break  
    
    # Print the name of the middle brother to standard output.
    print(middle_brother)

# Call the solve function to execute the main logic of the program.
solve()