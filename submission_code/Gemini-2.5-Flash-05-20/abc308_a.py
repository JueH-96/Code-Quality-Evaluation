import sys

def solve():
    # Read the input line and split it into strings, then convert to integers
    s_str = sys.stdin.readline().split()
    S = [int(x) for x in s_str]

    # Initialize a flag to track if all conditions are met
    is_valid = True

    # Iterate through the sequence from the first element to the last
    for i in range(len(S)):
        current_s = S[i]

        # Condition 2: S_i are all between 100 and 675, inclusive.
        if not (100 <= current_s <= 675):
            is_valid = False
            break # No need to check further if this condition is violated

        # Condition 3: S_i are all multiples of 25.
        if not (current_s % 25 == 0):
            is_valid = False
            break # No need to check further if this condition is violated

        # Condition 1: The sequence is monotonically non-decreasing.
        # This check applies from the second element (index 1) onwards.
        # We compare the current element with the previous one.
        if i > 0:
            if not (S[i-1] <= current_s):
                is_valid = False
                break # No need to check further if this condition is violated
    
    # Print the result based on the final state of the is_valid flag
    if is_valid:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the program
solve()