def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    K, G, M = map(int, input_data)

    glass = 0  # Water in the glass (initially 0)
    mug = 0    # Water in the mug (initially 0)

    for _ in range(K):
        # If the glass is full, discard its water.
        if glass == G:
            glass = 0
        # Otherwise, if the mug is empty, fill it with water.
        elif mug == 0:
            mug = M
        # Otherwise, transfer water from the mug to the glass.
        else:
            transfer = min(G - glass, mug)
            glass += transfer
            mug -= transfer

    print(glass, mug)

if __name__ == '__main__':
    main()