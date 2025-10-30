def main():
    import sys
    data = sys.stdin.readline().split()
    K = int(data[0])
    G = int(data[1])
    M = int(data[2])
    
    glass = 0
    mug = 0
    
    for _ in range(K):
        if glass == G:
            glass = 0
        elif mug == 0:
            mug = M
        else:
            transfer_amount = min(G - glass, mug)
            glass += transfer_amount
            mug -= transfer_amount
    
    print(f"{glass} {mug}")

if __name__ == "__main__":
    main()