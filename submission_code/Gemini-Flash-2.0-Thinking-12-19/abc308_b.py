def solve():
    n, m = map(int, input().split())
    colors_eaten = input().split()
    special_colors = input().split()
    prices = list(map(int, input().split()))
    p0 = prices[0]
    special_prices = {}
    for i in range(m):
        special_colors_i = special_colors[i]
        price_i = prices[i+1]
        special_prices[special_colors_i] = price_i
    
    total_price = 0
    for color in colors_eaten:
        if color in special_prices:
            total_price += special_prices[color]
        else:
            total_price += p0
            
    print(total_price)

if __name__ == '__main__':
    solve()