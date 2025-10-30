def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    dishes = []
    for i in range(n):
        dishes.append((a[i], b[i]))
    
    def calculate_dishes_eaten(ordered_dishes, x_limit, y_limit):
        current_sweetness = 0
        current_saltiness = 0
        count = 0
        for dish in ordered_dishes:
            current_sweetness += dish[0]
            current_saltiness += dish[1]
            count += 1
            if current_sweetness > x_limit or current_saltiness > y_limit:
                return count
        return count
        
    sorted_by_sweetness = sorted(dishes, key=lambda d: d[0], reverse=True)
    dishes_eaten_sweetness = calculate_dishes_eaten(sorted_by_sweetness, x, y)
    
    sorted_by_saltiness = sorted(dishes, key=lambda d: d[1], reverse=True)
    dishes_eaten_saltiness = calculate_dishes_eaten(sorted_by_saltiness, x, y)
    
    print(min(dishes_eaten_sweetness, dishes_eaten_saltiness))

if __name__ == "__main__":
    solve()