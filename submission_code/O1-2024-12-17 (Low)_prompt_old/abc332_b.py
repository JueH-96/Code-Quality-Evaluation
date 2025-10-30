def solve():
    import sys
    data = sys.stdin.read().strip().split()
    K = int(data[0])
    G = int(data[1])
    M = int(data[2])
    
    # Initialize the amounts of water in each container
    glass = 0  # current water in the glass
    mug = 0    # current water in the mug
    
    for _ in range(K):
        if glass == G:
            # If the glass is full, discard all the water from the glass
            glass = 0
        elif mug == 0:
            # If the mug is empty, fill the mug
            mug = M
        else:
            # Otherwise, transfer water from the mug to the glass
            transfer_amount = min(G - glass, mug)
            glass += transfer_amount
            mug -= transfer_amount
    
    # Print the results
    print(glass, mug)

# Call the solve() function
if __name__ == "__main__":
    solve()