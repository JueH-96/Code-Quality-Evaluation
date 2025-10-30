def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    
    # Read the colors of the plates Takahashi ate
    C = data[idx:idx+N]
    idx += N
    
    # Read the known colors and their associated prices
    D = data[idx:idx+M]
    idx += M
    
    # Read P_0, P_1, ..., P_M
    prices = list(map(int, data[idx:idx+M+1]))
    P0 = prices[0]
    P = prices[1:]
    
    # Create a dictionary to map known colors to their prices
    color_to_price = {}
    for i in range(M):
        color_to_price[D[i]] = P[i]
    
    # Calculate total price
    total_price = 0
    for color in C:
        total_price += color_to_price.get(color, P0)
    
    print(total_price)

if __name__ == "__main__":
    main()