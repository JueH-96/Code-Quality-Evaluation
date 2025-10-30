import sys

def main() -> None:
    # Read input
    data = sys.stdin.readline().strip().split()
    if not data:
        return
    K, G, M = map(int, data)

    glass = 0  # current volume in the glass
    mug   = 0  # current volume in the mug

    for _ in range(K):
        if glass == G:
            # 1. If the glass is full, discard its contents.
            glass = 0
        elif mug == 0:
            # 2. Otherwise, if the mug is empty, fill the mug.
            mug = M
        else:
            # 3. Otherwise, pour from the mug to the glass.
            transfer = min(G - glass, mug)
            glass   += transfer
            mug     -= transfer

    # Output the result
    print(glass, mug)

if __name__ == "__main__":
    main()