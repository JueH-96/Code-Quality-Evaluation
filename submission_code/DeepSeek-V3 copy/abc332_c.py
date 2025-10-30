def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Initialize the number of logo T-shirts needed
    logo_needed = 0
    # Initialize the number of plain T-shirts available
    plain_available = M
    # Initialize the number of logo T-shirts available
    logo_available = 0
    
    # Iterate through each day
    for i in range(N):
        if S[i] == '0':
            # Wash all T-shirts, so reset the available counts
            plain_available = M
            logo_available = logo_needed
        elif S[i] == '1':
            # Prefer to use plain T-shirts first
            if plain_available > 0:
                plain_available -= 1
            else:
                # Use a logo T-shirt
                if logo_available > 0:
                    logo_available -= 1
                else:
                    # Need to buy a new logo T-shirt
                    logo_needed += 1
                    logo_available += 1
                    logo_available -= 1
        elif S[i] == '2':
            # Must use a logo T-shirt
            if logo_available > 0:
                logo_available -= 1
            else:
                # Need to buy a new logo T-shirt
                logo_needed += 1
                logo_available += 1
                logo_available -= 1
    
    print(logo_needed)

if __name__ == "__main__":
    main()