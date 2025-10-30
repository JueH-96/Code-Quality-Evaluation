def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())

    total_length = sum(d)
    min_cost = float('inf')
    possible = False

    for num_l1 in range(k1 + 1):
        for num_l2 in range(k2 + 1):
            if num_l1 * l1 + num_l2 * l2 >= total_length:
                possible = True
                min_cost = min(min_cost, num_l1 * c1 + num_l2 * c2)

    if possible:
        print(min_cost)
    else:
        print(-1)

solve()