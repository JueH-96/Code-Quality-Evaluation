def solve():
    n, m = map(int, input().split())
    colors_eaten = input().split()
    special_colors = input().split()
    prices = list(map(int, input().split()))
    default_price = prices[0]
    special_prices = prices[1:]
    
    price_map = {}
    for i in range(m):
        price_map[special_colors[i]] = special_prices[i]
        
    total_price = 0
    for color in colors_eaten:
        if color in price_map:
            total_price += price_map[color]
        else:
            total_price += default_price
            
    print(total_price)

if __name__ == '__main__':
    solve()