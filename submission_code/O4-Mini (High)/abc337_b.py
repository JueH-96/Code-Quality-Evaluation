def main():
    import sys
    S = sys.stdin.readline().strip()
    # Check that S consists of zero or more 'A's, then zero or more 'B's, then zero or more 'C's.
    state = 0
    for ch in S:
        if ch == 'A':
            # Can only see 'A' in state 0
            if state != 0:
                print("No")
                return
        elif ch == 'B':
            # Transition to B-reading state
            if state <= 1:
                state = 1
            else:
                print("No")
                return
        elif ch == 'C':
            # Transition to C-reading state
            if state <= 2:
                state = 2
            else:
                print("No")
                return
        else:
            # Invalid character
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()