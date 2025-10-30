def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(tuple(map(int, input().split())))
    
    memo = {}
    
    def get_max_dishes(current_sweetness, current_saltiness, remaining_dish_indices):
        state = (current_sweetness, current_saltiness, tuple(sorted(remaining_dish_indices)))
        if state in memo:
            return memo[state]
        
        max_count = 0
        for i in range(len(remaining_dish_indices)):
            dish_index = remaining_dish_indices[i]
            sweetness, saltiness = dishes[dish_index]
            if current_sweetness + sweetness <= x and current_saltiness + saltiness <= y:
                next_remaining_indices = list(remaining_dish_indices)
                next_remaining_indices.pop(i)
                count = 1 + get_max_dishes(current_sweetness + sweetness, current_saltiness + saltiness, tuple(next_remaining_indices))
                max_count = max(max_count, count)
                
        memo[state] = max_count
        return max_count
        
    initial_indices = tuple(range(n))
    result = get_max_dishes(0, 0, initial_indices)
    print(result)

if __name__ == '__main__':
    solve()