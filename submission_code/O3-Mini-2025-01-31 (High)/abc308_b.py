def main():
    import sys
    input_data = sys.stdin.read().split()
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Next N entries are the colors of the sushi Takahashi ate
    sushi_colors = input_data[2:2+N]
    
    # Next M entries are the special colors
    special_colors = input_data[2+N:2+N+M]
    
    # Last M+1 entries are prices: first is P0, then P1,...,PM.
    prices = list(map(int, input_data[2+N+M:2+N+M+M+1]))
    
    # Create a dictionary to map special color to its price (P_i for i=1..M).
    price_map = {color: prices[i+1] for i, color in enumerate(special_colors)}
    
    total_price = 0
    default_price = prices[0]
    for color in sushi_colors:
        if color in price_map:
            total_price += price_map[color]
        else:
            total_price += default_price
            
    print(total_price)

if __name__ == '__main__':
    main()