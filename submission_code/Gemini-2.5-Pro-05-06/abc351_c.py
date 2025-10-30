import sys

def solve():
    # Read N, the number of balls/operations
    N = int(sys.stdin.readline())
    
    # Read the list of exponents A_i for each ball
    # A_values[i] is the exponent for the (i+1)-th ball
    A_values = list(map(int, sys.stdin.readline().split()))

    # The list 's' will simulate the sequence of balls, storing their exponents.
    # It functions as a stack, with operations on the right end.
    s = [] 

    # Process each ball one by one
    for current_exponent in A_values:
        # Add the current ball (its exponent) to the right end of the sequence
        s.append(current_exponent)

        # Repeatedly perform merging operations as long as conditions are met:
        # Condition 1: The sequence must have at least two balls (len(s) >= 2).
        # Condition 2: The rightmost two balls must have the same size,
        #              meaning their exponents are equal (s[-1] == s[-2]).
        while len(s) >= 2 and s[-1] == s[-2]:
            # If both conditions are true, merge the two balls.
            
            # The exponent of these two identical balls is s[-1].
            # Pop the rightmost ball and store its exponent.
            exp = s.pop() 
            
            # Pop the second rightmost ball (which had the same exponent 'exp').
            s.pop() 
            
            # Add a new ball to the right end of the sequence.
            # Its exponent is 'exp + 1' (representing size 2^(exp+1)).
            s.append(exp + 1)
        
        # The merging loop terminates if:
        # - The sequence has fewer than two balls (len(s) < 2), or
        # - The rightmost two balls have different sizes (s[-1] != s[-2]).
        
    # After all N operations, the number of balls remaining in the sequence
    # is the current length of the list 's'.
    print(len(s))

# Standard boilerplate to call the solve function when the script is executed
if __name__ == '__main__':
    solve()