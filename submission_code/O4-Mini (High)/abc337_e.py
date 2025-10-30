import sys

def main():
    data = sys.stdin.readline().strip()
    if not data:
        return
    N = int(data)
    # Determine minimal M such that 2^M >= N
    M = 0
    while (1 << M) < N:
        M += 1

    # Output number of friends (tests)
    print(M)
    sys.stdout.flush()

    # For each bit position i = 0..M-1, assign that bit-test to friend i+1
    # Friend i+1 tastes exactly those bottles whose (j-1) has bit i set.
    for i in range(M):
        # Build list of bottles for friend i
        bottles = []
        for j in range(1, N + 1):
            if ((j - 1) >> i) & 1:
                bottles.append(str(j))
        if bottles:
            print(len(bottles), " ".join(bottles))
        else:
            # No bottles for this friend
            print(0)
        sys.stdout.flush()

    # Read the judge's response string S of length M
    S = sys.stdin.readline().strip()
    # Reconstruct the spoiled bottle number from the bit pattern
    idx = 0
    for i, ch in enumerate(S):
        if ch == '1':
            idx |= (1 << i)
    # Our numbering was (j-1) -> bits, so add 1 back
    spoiled = idx + 1

    # Output the answer
    print(spoiled)
    sys.stdout.flush()

if __name__ == "__main__":
    main()