def main():
    import sys
    input = sys.stdin.readline

    # Read N and M
    N, M = map(int, input().split())
    # Read the list of plate colors Takahashi ate
    sushi_colors = input().split()
    # Read the list of color mappings for special prices
    special_colors = input().split()
    # Read the prices: first one is P0 and then P1..PM in order of special_colors
    prices = list(map(int, input().split()))
    
    P0 = prices[0]
    price_map = {}
    
    for i in range(M):
        color = special_colors[i]
        # Price for this color is prices[i+1]
        price_map[color] = prices[i+1]
    
    total_cost = 0
    for color in sushi_colors:
        if color in price_map:
            total_cost += price_map[color]
        else:
            total_cost += P0
            
    print(total_cost)

if __name__ == '__main__':
    main()