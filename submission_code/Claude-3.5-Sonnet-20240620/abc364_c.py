# YOUR CODE HERE
def min_dishes_eaten():
    N, X, Y = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    dishes = sorted(zip(A, B), key=lambda x: max(x[0], x[1]))

    total_sweetness = 0
    total_saltiness = 0
    dishes_eaten = 0

    for sweetness, saltiness in dishes:
        if total_sweetness > X or total_saltiness > Y:
            break
        total_sweetness += sweetness
        total_saltiness += saltiness
        dishes_eaten += 1

    return dishes_eaten

print(min_dishes_eaten())