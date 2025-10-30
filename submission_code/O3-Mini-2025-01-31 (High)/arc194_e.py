def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Read inputs:
    N = int(data[0])
    X = int(data[1])
    Y = int(data[2])
    S = data[3].strip()
    T = data[4].strip()

    # If N < X+Y then no operation is allowed; check direct equality.
    if N < X + Y:
        sys.stdout.write("Yes" if S == T else "No")
        return

    # The total count of ones must match.
    if S.count('1') != T.count('1'):
        sys.stdout.write("No")
        return

    # Create an array of counts for each residue mod X.
    countsS = [0] * X
    countsT = [0] * X

    # (We use 0-index positions; note that a move of ±X does not change i mod X.)
    for i, ch in enumerate(S):
        if ch == '1':
            countsS[i % X] += 1
    for i, ch in enumerate(T):
        if ch == '1':
            countsT[i % X] += 1

    # If the counts in each residue class match then S can be rearranged to T.
    if countsS == countsT:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")
    
if __name__ == '__main__':
    main()