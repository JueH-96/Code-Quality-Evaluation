# YOUR CODE HERE
def solve():
    n, x, y = map(int, input().split())
    dishes = []
    for _ in range(n):
        dishes.append(list(map(int, input().split())))

    max_eaten = 0

    import itertools
    for order in itertools.permutations(range(n)):
        current_sweetness = 0
        current_saltiness = 0
        eaten_count = 0
        for i in order:
            current_sweetness += dishes[i][0]
            current_saltiness += dishes[i][1]
            if current_sweetness <= x and current_saltiness <= y:
                eaten_count += 1
            else:
                break
        max_eaten = max(max_eaten, eaten_count)
    
    print(max_eaten)

solve()