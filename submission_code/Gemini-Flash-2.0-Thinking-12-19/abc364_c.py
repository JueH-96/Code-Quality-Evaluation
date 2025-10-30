def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    dishes = []
    for i in range(n):
        dishes.append({'a': a[i], 'b': b[i], 'index': i})

    def calculate_dishes_eaten(order):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten_count = 0
        for dish_index in order:
            current_sweetness += dishes[dish_index]['a']
            current_saltiness += dishes[dish_index]['b']
            dishes_eaten_count += 1
            if current_sweetness > x or current_saltiness > y:
                return dishes_eaten_count
        return dishes_eaten_count

    min_dishes = n

    # Sort by sweetness descending
    sorted_indices_a = sorted(range(n), key=lambda i: dishes[i]['a'], reverse=True)
    dishes_eaten_a = calculate_dishes_eaten(sorted_indices_a)
    min_dishes = min(min_dishes, dishes_eaten_a)

    # Sort by saltiness descending
    sorted_indices_b = sorted(range(n), key=lambda i: dishes[i]['b'], reverse=True)
    dishes_eaten_b = calculate_dishes_eaten(sorted_indices_b)
    min_dishes = min(min_dishes, dishes_eaten_b)

    import itertools
    min_dishes_brute = n + 1
    for p in itertools.permutations(range(n)):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten_count = 0
        for i in range(n):
            dish_index = p[i]
            current_sweetness += dishes[dish_index]['a']
            current_saltiness += dishes[dish_index]['b']
            dishes_eaten_count += 1
            if current_sweetness > x or current_saltiness > y:
                min_dishes_brute = min(min_dishes_brute, dishes_eaten_count)
                break
        else:
            min_dishes_brute = min(min_dishes_brute, n)

    if min_dishes_brute <= n:
        print(min_dishes_brute)
    else:
        print(min_dishes)


solve()