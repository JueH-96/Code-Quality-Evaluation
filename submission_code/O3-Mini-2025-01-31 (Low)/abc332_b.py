def main():
    import sys
    input = sys.stdin.readline
    
    # read inputs
    data = input().split()
    if not data:
        return
    K, G, M = map(int, data)
    
    # initial amounts: both glass and mug are empty
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:
            # if glass is full, discard all water from the glass
            glass = 0
        elif mug == 0:
            # if the mug is empty, fill the mug
            mug = M
        else:
            # otherwise transfer water from the mug to the glass
            transfer = min(G - glass, mug)
            glass += transfer
            mug -= transfer
    
    # print milliliters in the glass and mug
    print(glass, mug)

if __name__ == '__main__':
    main()