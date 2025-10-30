def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    K = int(input_data[0])
    G = int(input_data[1])
    M = int(input_data[2])
    
    # Initially both are empty.
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:
            # Discard water from the glass
            glass = 0
        elif mug == 0:
            # Fill the mug with water
            mug = M
        else:
            # Transfer water from mug to glass until the mug is empty or glass is filled.
            transfer = min(mug, G - glass)
            glass += transfer
            mug -= transfer
    
    # Print the results: amount in glass and mug separated by a space.
    print(glass, mug)

if __name__ == '__main__':
    main()