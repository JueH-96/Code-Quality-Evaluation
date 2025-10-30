def solve():
    n, m, l = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    forbidden_pairs = set()
    for _ in range(l):
        u, v = map(int, input().split())
        forbidden_pairs.add((u, v))
    
    main_dishes = []
    for i in range(n):
        main_dishes.append({'price': a[i], 'index': i + 1})
    
    side_dishes = []
    for i in range(m):
        side_dishes.append({'price': b[i], 'index': i + 1})
        
    max_price = 0
    
    for main_dish in main_dishes:
        for side_dish in side_dishes:
            main_dish_index = main_dish['index']
            side_dish_index = side_dish['index']
            if (main_dish_index, side_dish_index) not in forbidden_pairs:
                current_price = main_dish['price'] + side_dish['price']
                if current_price > max_price:
                    max_price = current_price
                    
    print(max_price)

if __name__ == '__main__':
    solve()