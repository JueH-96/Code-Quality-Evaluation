def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N_M = data[0].split()
    N = int(N_M[0])
    M = int(N_M[1])
    
    # Read the colors of the N sushi plates
    C = data[1].split()
    
    # Read the M special colors
    D = data[2].split()
    
    # Read the prices P_0 to P_M
    P = list(map(int, data[3].split()))
    
    # Create a dictionary mapping from special color to its price
    special_prices = {D[i]: P[i+1] for i in range(M)}
    
    # P_0 is the default price
    P0 = P[0]
    
    # Calculate the total price
    total = 0
    for color in C:
        if color in special_prices:
            total += special_prices[color]
        else:
            total += P0
    
    # Print the total amount
    print(total)

if __name__ == "__main__":
    main()