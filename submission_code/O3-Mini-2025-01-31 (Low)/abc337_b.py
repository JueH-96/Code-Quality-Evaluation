def main():
    import sys
    S = sys.stdin.read().strip()
    
    # We want the string S to be in the form: A...A B...B C...C
    # where each segment possibly empty.
    # Let's use a state machine as follows:
    # state 0: we expect only 'A'. Once we see a 'B' transition to state 1.
    # state 1: we expect only 'B'. Once we see a 'C' transition to state 2.
    # state 2: we expect only 'C'. Anything else is an error.
    
    state = 0
    for ch in S:
        if state == 0:
            if ch == 'A':
                continue
            elif ch == 'B':
                state = 1
            elif ch == 'C':
                # if the first non-A char is C, but we need B segment (it can be empty),
                # So we transition to state 2 directly. But that's allowed.
                state = 2
            else:
                print("No")
                return
        elif state == 1:
            if ch == 'B':
                continue
            elif ch == 'C':
                state = 2
            else:
                # if we see an A in state 1, it's invalid.
                print("No")
                return
        elif state == 2:
            if ch == 'C':
                continue
            else:
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()