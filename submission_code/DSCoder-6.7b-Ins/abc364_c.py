import sys
from operator import itemgetter

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]

    dishes.sort(key=itemgetter(1), reverse=True)
    dishes.sort(key=itemgetter(0), reverse=True)

    total_sweetness = 0
    total_saltiness = 0
    dishes_eaten = 0

    for sweetness, saltiness in dishes:
        if total_sweetness + sweetness <= X and total_saltiness + saltiness <= Y:
            total_sweetness += sweetness
            total_saltiness += saltiness
            dishes_eaten += 1
        else:
            break

    print(dishes_eaten)

solve()