def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(list(map(int, input().split())))
    
    max_dishes_eaten = 0
    
    import itertools
    
    dish_indices = list(range(n))
    
    for permutation_indices in itertools.permutations(dish_indices):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten_count = 0
        for dish_index in permutation_indices:
            sweetness, saltiness = dishes[dish_index]
            if current_sweetness + sweetness <= x and current_saltiness + saltiness <= y:
                current_sweetness += sweetness
                current_saltiness += saltiness
                dishes_eaten_count += 1
            else:
                break
        max_dishes_eaten = max(max_dishes_eaten, dishes_eaten_count)
        
    print(max_dishes_eaten)

if __name__ == '__main__':
    solve()