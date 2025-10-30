def main():
    import sys
    data = sys.stdin.readline().split()
    K, G, M = map(int, data)
    glass = 0
    mug = 0

    for _ in range(K):
        if glass == G:
            # discard all water from the glass
            glass = 0
        elif mug == 0:
            # fill the mug
            mug = M
        else:
            # transfer water from the mug to the glass
            t = min(mug, G - glass)
            mug -= t
            glass += t

    print(glass, mug)

if __name__ == "__main__":
    main()