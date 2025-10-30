def main():
    import sys
    data = sys.stdin.read().strip().split()
    K = int(data[0])  # We know K = 1 from the problem statement
    S = data[1]
    T = data[2]

    # If they are already the same, 0 operations needed.
    if S == T:
        print("Yes")
        return

    # If length difference is more than 1, impossible with just one operation.
    if abs(len(S) - len(T)) > 1:
        print("No")
        return

    # Case 1: Same length, check if they differ in at most one character (replacement).
    if len(S) == len(T):
        mismatch = 0
        for c1, c2 in zip(S, T):
            if c1 != c2:
                mismatch += 1
                if mismatch > 1:
                    print("No")
                    return
        print("Yes")
        return

    # Helper function: checks if t is s with exactly one extra character
    # at some position (so s can be turned into t by one insertion).
    def isOneInsertAway(s, t):
        # Expect len(t) == len(s) + 1
        i, j = 0, 0
        used_skip = False
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if used_skip:
                    return False
                used_skip = True
                j += 1
        return True

    # Case 2: T is S + 1 character (insertion in S).
    if len(T) == len(S) + 1:
        if isOneInsertAway(S, T):
            print("Yes")
        else:
            print("No")
        return

    # Case 3: S is T + 1 character (deletion in S).
    if len(S) == len(T) + 1:
        # If S can become T by deleting one char, T should be "one insert away" from T's perspective.
        if isOneInsertAway(T, S):
            print("Yes")
        else:
            print("No")
        return

# Do not forget to call main
main()