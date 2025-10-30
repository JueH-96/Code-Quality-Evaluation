# YOUR CODE HERE
def solve():
    n, x, y = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    dishes = []
    for i in range(n):
        dishes.append((a[i], b[i]))

    ans = float('inf')
    import itertools
    for order in itertools.permutations(dishes):
        total_sweetness = 0
        total_saltiness = 0
        count = 0
        for sweet, salty in order:
            total_sweetness += sweet
            total_saltiness += salty
            count += 1
            if total_sweetness > x or total_saltiness > y:
                ans = min(ans, count)
                break
        else:
            ans = min(ans, count)
    
    print(ans)

solve()