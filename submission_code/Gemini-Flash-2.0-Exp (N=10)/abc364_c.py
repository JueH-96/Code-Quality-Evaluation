import itertools

def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    min_dishes = float('inf')

    for perm in itertools.permutations(range(n)):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten = 0
        
        for i in perm:
            current_sweetness += a[i]
            current_saltiness += b[i]
            dishes_eaten += 1
            
            if current_sweetness > x or current_saltiness > y:
                min_dishes = min(min_dishes, dishes_eaten)
                break
        else:
            min_dishes = min(min_dishes, dishes_eaten)

    print(min_dishes)

solve()