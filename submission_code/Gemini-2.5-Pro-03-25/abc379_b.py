# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Read N and K from the first line of standard input
    line1 = sys.stdin.readline().split()
    n = int(line1[0]) # Number of teeth
    k = int(line1[1]) # Number of consecutive healthy teeth needed

    # Read the string S representing teeth condition from the second line
    s = sys.stdin.readline().strip() # String of 'O' (healthy) and 'X' (cavity)

    # Initialize the total count of strawberries Takahashi can eat
    strawberry_count = 0
    
    # Initialize the count of currently observed consecutive healthy ('O') teeth
    consecutive_o_count = 0
    
    # Iterate through each tooth position from left to right (index 0 to n-1)
    for i in range(n):
        if s[i] == 'O':
            # If the current tooth is healthy ('O'), increment the consecutive count
            consecutive_o_count += 1
        else:
            # If the current tooth has a cavity ('X'), the sequence of healthy teeth is broken.
            # Reset the consecutive count to 0.
            consecutive_o_count = 0 
            
        # Check if we have just completed a sequence of exactly K consecutive healthy teeth
        # This condition becomes true when the consecutive count reaches K.
        if consecutive_o_count == k:
            # If yes, Takahashi can eat one strawberry using these K teeth.
            strawberry_count += 1
            
            # After eating a strawberry, those K teeth develop cavities.
            # We model this by resetting the consecutive count of healthy teeth to 0.
            # This ensures that these K teeth cannot be immediately reused or contribute
            # to forming another overlapping sequence starting within them.
            # Example: If S="OOOO" and K=3, we find 'OOO' ending at index 2.
            # We count 1 strawberry and reset consecutive_o_count to 0.
            # When processing index 3 ('O'), consecutive_o_count becomes 1.
            # We don't find another K=3 sequence, which is correct.
            consecutive_o_count = 0 
            
    # After iterating through all the teeth, print the total number of strawberries eaten.
    print(strawberry_count)

# Call the solve function to execute the main logic of the program
solve()