def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    # Number of plain T-shirts available
    plain_tshirts = M
    # Number of logo T-shirts needed
    logo_tshirts_needed = 0
    
    # Track the number of logo T-shirts in use
    logo_tshirts_in_use = 0
    
    # Iterate over each day
    for i in range(N):
        if S[i] == '0':
            # Wash day, reset the in-use counters
            logo_tshirts_in_use = 0
        elif S[i] == '1':
            # Needs a plain or logo T-shirt
            if plain_tshirts > 0:
                # Use a plain T-shirt if available
                plain_tshirts -= 1
            else:
                # Use a logo T-shirt if no plain T-shirts are available
                logo_tshirts_in_use += 1
        elif S[i] == '2':
            # Needs a logo T-shirt
            logo_tshirts_in_use += 1
        
        # Check if we need to buy more logo T-shirts
        if logo_tshirts_in_use > logo_tshirts_needed:
            logo_tshirts_needed = logo_tshirts_in_use
    
    # Calculate the number of logo T-shirts to buy
    logo_tshirts_to_buy = max(0, logo_tshirts_needed - plain_tshirts)
    
    # Output the result
    print(logo_tshirts_to_buy)

if __name__ == "__main__":
    main()