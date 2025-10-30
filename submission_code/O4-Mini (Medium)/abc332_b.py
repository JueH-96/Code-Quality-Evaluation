def main():
    import sys
    data = sys.stdin.read().split()
    K, G, M = map(int, data)
    glass = 0
    mug = 0
    for _ in range(K):
        if glass == G:
            # If the glass is full, discard it.
            glass = 0
        elif mug == 0:
            # If the mug is empty, fill it.
            mug = M
        else:
            # Otherwise, transfer water from the mug to the glass.
            amount = min(G - glass, mug)
            glass += amount
            mug -= amount
    # Output the final amounts.
    print(glass, mug)

# Call main to execute the program.
main()