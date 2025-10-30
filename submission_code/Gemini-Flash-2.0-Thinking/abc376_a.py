def solve():
    n, c = map(int, input().split())
    t = list(map(int, input().split()))

    candies = 0
    last_received_time = None

    for i in range(n):
        if i == 0:
            candies += 1
            last_received_time = t[i]
        else:
            if last_received_time is None or t[i] - last_received_time >= c:
                candies += 1
                last_received_time = t[i]

    print(candies)

solve()