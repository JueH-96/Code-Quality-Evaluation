import itertools

def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    min_dishes = n + 1

    for perm in itertools.permutations(range(n)):
        sweetness = 0
        saltiness = 0
        dishes_eaten = 0

        for i in perm:
            sweetness += a[i]
            saltiness += b[i]
            dishes_eaten += 1

            if sweetness > x or saltiness > y:
                min_dishes = min(min_dishes, dishes_eaten)
                break
        else:
            min_dishes = min(min_dishes, dishes_eaten)

    print(min_dishes)

solve()