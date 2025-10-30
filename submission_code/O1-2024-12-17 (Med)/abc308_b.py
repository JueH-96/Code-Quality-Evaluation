def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    
    # The next N strings are C_i
    C = data[2:2+N]
    
    # The next M strings are D_i
    D = data[2+N:2+N+M]
    
    # The next M+1 integers are P_0, P_1, ..., P_M
    prices = list(map(int, data[2+N+M:] ))
    P0 = prices[0]
    # Build a lookup for color -> price
    color_to_price = {}
    for i in range(M):
        color_to_price[D[i]] = prices[i+1]
    
    # Calculate the sum
    total_cost = 0
    for color in C:
        if color in color_to_price:
            total_cost += color_to_price[color]
        else:
            total_cost += P0
    
    print(total_cost)

# Do not forget to call main!
if __name__ == "__main__":
    main()