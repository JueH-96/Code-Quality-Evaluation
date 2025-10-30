def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    c_list = data[1].split()
    d_list = data[2].split()
    prices = list(map(int, data[3].split()))
    
    base_price = prices[0]
    color_to_price = {}
    for idx in range(m):
        color = d_list[idx]
        color_to_price[color] = prices[idx + 1]
    
    total = 0
    for color in c_list:
        if color in color_to_price:
            total += color_to_price[color]
        else:
            total += base_price
    
    print(total)

if __name__ == "__main__":
    main()