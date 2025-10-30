def solve():
    n, x = map(int, input().split())
    foods = []
    for _ in range(n):
        foods.append(list(map(int, input().split())))

    max_min_vitamin = 0
    for i in range(1 << n):
        calories = 0
        vitamins = [0, 0, 0]
        
        for j in range(n):
            if (i >> j) & 1:
                calories += foods[j][2]
                vitamins[foods[j][0] - 1] += foods[j][1]
        
        if calories <= x:
            min_vitamin = min(vitamins)
            max_min_vitamin = max(max_min_vitamin, min_vitamin)

    print(max_min_vitamin)

solve()