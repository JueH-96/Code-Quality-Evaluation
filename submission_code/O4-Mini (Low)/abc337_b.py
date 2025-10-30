def main():
    import sys

    S = sys.stdin.readline().strip()
    # We want to check if S is of the form A* B* C*
    # i.e. any number of A's, then any number of B's, then any number of C's, and nothing else.
    
    # State 0: still reading A's
    # State 1: reading B's
    # State 2: reading C's
    state = 0
    for ch in S:
        if state == 0:
            if ch == 'A':
                continue
            elif ch == 'B':
                state = 1
            elif ch == 'C':
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
                # ch is 'A' or something else
                print("No")
                return
        else:  # state == 2
            if ch != 'C':
                # only C's allowed now
                print("No")
                return

    # If we complete the loop without invalid transitions, it's valid
    print("Yes")

if __name__ == "__main__":
    main()