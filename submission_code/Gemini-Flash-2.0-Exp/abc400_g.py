def solve():
    N, K = map(int, input().split())
    cakes = []
    for _ in range(N):
        cakes.append(list(map(int, input().split())))

    import itertools

    max_price = 0
    for combination in itertools.combinations(range(N), 2 * K):
        current_price = 0
        pairs = []
        for i in range(K):
            pairs.append((combination[2 * i], combination[2 * i + 1]))

        for pair in pairs:
            cake1_index = pair[0]
            cake2_index = pair[1]
            
            price = max(cakes[cake1_index][0] + cakes[cake2_index][0],
                        cakes[cake1_index][1] + cakes[cake2_index][1],
                        cakes[cake1_index][2] + cakes[cake2_index][2])
            current_price += price
        
        max_price = max(max_price, current_price)

    print(max_price)

T = int(input())
for _ in range(T):
    solve()