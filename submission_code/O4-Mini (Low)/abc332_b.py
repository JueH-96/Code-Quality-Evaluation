def main():
    import sys
    data = sys.stdin.read().strip().split()
    K = int(data[0])
    G = int(data[1])
    M = int(data[2])

    glass = 0
    mug = 0

    for _ in range(K):
        if glass == G:
            # Discard all water from the glass
            glass = 0
        elif mug == 0:
            # Fill the mug
            mug = M
        else:
            # Transfer from mug to glass
            transfer = min(mug, G - glass)
            mug -= transfer
            glass += transfer

    print(glass, mug)

if __name__ == "__main__":
    main()