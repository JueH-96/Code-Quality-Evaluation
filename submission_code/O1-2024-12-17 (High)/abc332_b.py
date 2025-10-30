def main():
    import sys
    data = sys.stdin.read().strip().split()
    K, G, M = map(int, data)
    
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:  # If the glass is full, discard its water
            glass = 0
        elif mug == 0:  # If the mug is empty, fill it
            mug = M
        else:  # Otherwise, transfer from mug to glass
            space = G - glass
            if mug <= space:
                glass += mug
                mug = 0
            else:
                mug -= space
                glass = G
    
    print(glass, mug)

main()