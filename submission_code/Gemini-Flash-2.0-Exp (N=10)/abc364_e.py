import itertools

def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(list(map(int, input().split())))

    max_dishes_eaten = 0
    for perm in itertools.permutations(dishes):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten = 0
        for dish in perm:
            if current_sweetness + dish[0] <= x and current_saltiness + dish[1] <= y:
                current_sweetness += dish[0]
                current_saltiness += dish[1]
                dishes_eaten += 1
            else:
                break
        max_dishes_eaten = max(max_dishes_eaten, dishes_eaten)
    print(max_dishes_eaten)

solve()