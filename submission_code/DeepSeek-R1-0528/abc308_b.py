def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    plate_colors = data[1].split()
    dish_colors = data[2].split()
    prices = list(map(int, data[3].split()))
    
    default_price = prices[0]
    dish_dict = {}
    for i in range(m):
        dish = dish_colors[i]
        dish_dict[dish] = prices[i + 1]
    
    total = 0
    for color in plate_colors:
        if color in dish_dict:
            total += dish_dict[color]
        else:
            total += default_price
            
    print(total)

if __name__ == "__main__":
    main()