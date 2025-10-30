import sys

# Function to solve the problem
def solve():
    """
    Reads a string S from standard input and determines if it is an Extended ABC string.
    An Extended ABC string consists of zero or more 'A's, followed by zero or more 'B's,
    followed by zero or more 'C's.
    Prints "Yes" if S is an Extended ABC string, otherwise prints "No".
    """
    # Read the input string from standard input
    s = sys.stdin.readline().strip()
    
    # State variable: tracks the highest character type encountered so far in the correct sequence.
    # 0: Corresponds to the 'A' phase (or initial state before any character).
    #    Only 'A's are expected or allowed to continue this phase. 'B' transitions to state 1, 'C' transitions to state 2.
    # 1: Corresponds to the 'B' phase (must follow the 'A' phase or be the start if no 'A's).
    #    Only 'B's are expected or allowed to continue this phase. 'C' transitions to state 2. 'A' is invalid.
    # 2: Corresponds to the 'C' phase (must follow 'A' and/or 'B' phases or be the start if no 'A's or 'B's).
    #    Only 'C's are expected or allowed to continue this phase. 'A' or 'B' is invalid.
    current_char_type_state = 0 

    # Iterate through each character of the string
    for char in s:
        if char == 'A':
            # If 'A' is encountered, it must be during the 'A' phase (state 0).
            # If we have already seen 'B' or 'C' (state > 0), encountering 'A' violates the A*B*C* pattern.
            if current_char_type_state > 0:
                print("No")
                return
            # If currently in state 0 (A phase or initial), seeing 'A' means we stay in state 0.
            # The state variable `current_char_type_state` remains 0.
        
        elif char == 'B':
            # If 'B' is encountered, it must be during the 'A' phase (state 0, transitioning to B) 
            # or the 'B' phase (state 1, continuing B).
            # If we have already seen 'C' (state is 2), encountering 'B' violates the A*B*C* pattern.
            if current_char_type_state > 1:
                print("No")
                return
            # If seeing 'B' is valid (current state is 0 or 1), we transition to or remain in state 1 (B phase).
            current_char_type_state = 1
            
        elif char == 'C':
            # If 'C' is encountered, it can validly follow 'A's (state 0), 'B's (state 1), or 'C's (state 2).
            # It can also be the first character if the string starts with 'C'.
            # There's no state violation check needed based on the *previous* state for 'C' itself, 
            # because 'C' is the final allowed character type in the sequence.
            # Any character following 'C' that is not 'C' would be caught by the checks for 'A' and 'B'.
            # Upon seeing 'C', we transition to or remain in state 2 (C phase).
            current_char_type_state = 2
            
        else:
            # This case should theoretically not be reached given the problem constraints 
            # that S contains only 'A', 'B', 'C'.
            # Included as a safeguard against unexpected input.
            print("No") 
            return

    # If the loop completes without returning "No", it means the string successfully
    # adheres to the A*B*C* pattern (zero or more A's, followed by zero or more B's, 
    # followed by zero or more C's).
    print("Yes")

# Main execution block to run the solver function
if __name__ == '__main__':
    solve()