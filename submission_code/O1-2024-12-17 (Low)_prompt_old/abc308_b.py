def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    
    # Read the list of plate colors that Takahashi ate
    C = data[2:2+N]
    
    # Read the list of colors with defined prices
    D = data[2+N:2+N+M]
    
    # Read the prices: P0, then P1..PM
    prices = list(map(int, data[2+N+M:2+N+M+(M+1)]))
    P0 = prices[0]
    P = prices[1:]
    
    # Create a dictionary mapping color -> price
    color_price_map = {}
    for i, color in enumerate(D):
        color_price_map[color] = P[i]
    
    total_amount = 0
    # Calculate total price
    for color in C:
        if color in color_price_map:
            total_amount += color_price_map[color]
        else:
            total_amount += P0
    
    print(total_amount)

def main():
    solve()

if __name__ == "__main__":
    main()