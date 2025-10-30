import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    dishes.sort(key=lambda x: (x[0]+x[1], x[0]))

    total_sweetness = 0
    total_saltiness = 0
    count = 0

    for sweetness, saltiness in dishes:
        if total_sweetness + sweetness <= X and total_saltiness + saltiness <= Y:
            total_sweetness += sweetness
            total_saltiness += saltiness
            count += 1

    print(count)

solve()