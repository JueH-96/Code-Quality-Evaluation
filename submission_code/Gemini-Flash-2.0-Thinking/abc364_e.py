from itertools import permutations

def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(tuple(map(int, input().split())))

    max_dishes_eaten = 0

    for perm in permutations(range(n)):
        current_sweetness = 0
        current_saltiness = 0
        dishes_eaten_count = 0
        for i in perm:
            sweetness, saltiness = dishes[i]
            if current_sweetness + sweetness <= x and current_saltiness + saltiness <= y:
                current_sweetness += sweetness
                current_saltiness += saltiness
                dishes_eaten_count += 1
            else:
                break
        max_dishes_eaten = max(max_dishes_eaten, dishes_eaten_count)

    print(max_dishes_eaten)

if __name__ == "__main__":
    solve()