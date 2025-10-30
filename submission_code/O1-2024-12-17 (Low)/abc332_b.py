def main():
    import sys
    data = sys.stdin.read().strip().split()
    K, G, M = map(int, data)

    glass = 0  # current water in glass
    mug = 0    # current water in mug

    for _ in range(K):
        # If the glass is full (glass == G), discard it
        if glass == G:
            glass = 0
        # Otherwise, if the mug is empty, fill the mug
        elif mug == 0:
            mug = M
        # Otherwise, transfer from the mug to the glass
        else:
            # Calculate how much the glass needs to be full
            needed = G - glass
            if mug <= needed:
                # All water in the mug goes into the glass
                glass += mug
                mug = 0
            else:
                # Mug still has some left after filling the glass
                mug -= needed
                glass = G
    
    print(glass, mug)

if __name__ == "__main__":
    main()