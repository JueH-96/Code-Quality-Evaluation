def main():
    import sys
    # Read the input string and remove extra whitespace/newlines
    S = sys.stdin.readline().strip()
    
    # We want to determine if S is of the form A*B*C*, i.e.
    # zero or more A's, followed by zero or more B's, followed by zero or more C's.
    # This exactly means that S can be split as S = S_A + S_B + S_C where
    # S_A consists only of A's, S_B only of B's, and S_C only of C's. Note that any of these
    # parts may be empty.
    
    state = 'A'
    valid = True
    for ch in S:
        if state == 'A':
            if ch == 'A':
                continue
            elif ch == 'B':
                state = 'B'
            elif ch == 'C':
                state = 'C'
            else:
                valid = False
                break
        elif state == 'B':
            if ch == 'B':
                continue
            elif ch == 'C':
                state = 'C'
            else:  # encountered 'A' in state 'B': invalid.
                valid = False
                break
        elif state == 'C':
            if ch == 'C':
                continue
            else:
                valid = False
                break
                
    # Output the result based on validity.
    sys.stdout.write("Yes" if valid else "No")

# Ensure main() is executed when the script is run.
if __name__ == '__main__':
    main()