# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    K, G, M = map(int, data.split())
    
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:
            # Discard all the water from the glass
            glass = 0
        elif mug == 0:
            # Fill the mug with water
            mug = M
        else:
            # Transfer water from the mug to the glass
            transfer_amount = min(G - glass, mug)
            glass += transfer_amount
            mug -= transfer_amount
    
    print(glass, mug)

if __name__ == "__main__":
    main()