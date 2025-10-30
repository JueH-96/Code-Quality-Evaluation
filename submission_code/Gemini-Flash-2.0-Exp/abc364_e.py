from itertools import permutations

def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(list(map(int, input().split())))

    max_dishes = 0
    for perm in permutations(dishes):
        sweetness = 0
        saltiness = 0
        num_dishes = 0
        for dish in perm:
            if sweetness + dish[0] <= x and saltiness + dish[1] <= y:
                sweetness += dish[0]
                saltiness += dish[1]
                num_dishes += 1
            else:
                break
        max_dishes = max(max_dishes, num_dishes)

    print(max_dishes)

solve()