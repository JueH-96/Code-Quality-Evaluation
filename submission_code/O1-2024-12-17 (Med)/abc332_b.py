def main():
    import sys
    data = sys.stdin.read().strip().split()
    K, G, M = map(int, data)

    glass = 0
    mug = 0

    for _ in range(K):
        if glass == G:
            # Discard all the water from the glass
            glass = 0
        elif mug == 0:
            # Fill the mug
            mug = M
        else:
            # Transfer from mug to glass until glass is full or mug is empty
            diff = G - glass
            if mug <= diff:
                # All water from mug goes into the glass
                glass += mug
                mug = 0
            else:
                # Glass becomes full, some water remains in mug
                glass = G
                mug -= diff

    print(glass, mug)

# Do not remove the following call
main()